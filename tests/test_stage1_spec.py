import hashlib
import json
import math
import unittest
from pathlib import Path

import torch

from pace_stage1.config import STAGE1_CONFIG, ppo_train_cfg
from pace_stage1.runner_bridge import apply_iteration_schedules
from pace_stage1.semantics import (
    BatchedPACEActuator,
    build_actor_observation,
    build_critic_observation,
    command_resample_mask,
    compute_reward_terms,
    entropy_schedule,
    make_target_adapter,
    penalty_schedule,
    reward_manifest,
    reset_true_joint_position,
    sample_commands,
    timeout_bootstrap,
    timeout_mask,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "provenance" / "stage1_manifest.json").read_text(encoding="utf-8"))
AUTHORIZATION = json.loads(
    (ROOT / "provenance" / "stage1_ppo_authorization.json").read_text(encoding="utf-8")
)


class Stage1ObservationSpecTest(unittest.TestCase):
    def _components(self, count=2):
        return (
            torch.ones(count, 3),
            torch.ones(count, 3) * 2,
            torch.ones(count, 3) * 3,
            torch.ones(count, 4) * 4,
            torch.ones(count, 12) * 5,
            torch.ones(count, 12) * 6,
            torch.ones(count, 12) * 7,
        )

    def test_actor_dimension_order_and_scaling(self):
        obs = build_actor_observation(*self._components(), add_noise=False)
        self.assertEqual(tuple(obs.shape), (2, 48))
        torch.testing.assert_close(obs[0, 0:3], torch.full((3,), 2.0))
        torch.testing.assert_close(obs[0, 3:6], torch.full((3,), 0.5))
        torch.testing.assert_close(obs[0, 9:12], torch.tensor([8.0, 8.0, 1.0]))
        torch.testing.assert_close(obs[0, 12:24], torch.full((12,), 5.0))
        torch.testing.assert_close(obs[0, 24:36], torch.full((12,), 0.3))
        torch.testing.assert_close(obs[0, 36:48], torch.full((12,), 7.0))
        self.assertEqual(STAGE1_CONFIG.observation.order[-1], "previous_policy_action[12]")

    def test_noise_excludes_commands_and_previous_action(self):
        generator = torch.Generator().manual_seed(10)
        clean = build_actor_observation(*self._components(), add_noise=False)
        noisy = build_actor_observation(*self._components(), add_noise=True, generator=generator)
        torch.testing.assert_close(clean[:, 9:12], noisy[:, 9:12])
        torch.testing.assert_close(clean[:, 36:48], noisy[:, 36:48])
        self.assertGreater(torch.max(torch.abs(clean[:, :9] - noisy[:, :9])).item(), 0.0)

    def test_critic_dimension_and_order(self):
        actor = torch.zeros(2, 48)
        critic = build_critic_observation(
            actor,
            torch.ones(2, 3),
            torch.ones(2, 3) * 2,
            torch.tensor([0.5, 0.7]),
            torch.tensor([[True, False, True, False], [False] * 4]),
            torch.zeros(2, 294),
        )
        self.assertEqual(tuple(critic.shape), (2, 353))
        torch.testing.assert_close(critic[:, 48:51], torch.ones(2, 3))
        torch.testing.assert_close(critic[:, 54], torch.tensor([0.5, 0.7]))
        self.assertEqual(sum(size for _, size in STAGE1_CONFIG.observation.critic_layout), 353)


class Stage1ActionAndActuatorTest(unittest.TestCase):
    def test_action_order_scale_default_and_timing(self):
        cfg = STAGE1_CONFIG.action
        self.assertEqual(cfg.dim, 12)
        self.assertEqual(cfg.joint_order[0:3], ("LF_HAA", "LF_HFE", "LF_KFE"))
        self.assertEqual(cfg.joint_order[3:6], ("RF_HAA", "RF_HFE", "RF_KFE"))
        self.assertEqual(cfg.policy_dt_s, 0.01)
        adapter = make_target_adapter()
        action = torch.zeros(1, 12)
        action[0, 0] = 1.0
        q0 = torch.tensor(cfg.default_joint_pose_rad).reshape(1, 12)
        lower = torch.full_like(q0, -2.0)
        upper = torch.full_like(q0, 2.0)
        target = adapter(action, q0, lower, upper)
        self.assertAlmostEqual(target[0, 0].item(), 0.5)
        torch.testing.assert_close(target[0, 1:], q0[0, 1:])

    def test_batched_actuator_inherits_frozen_step_and_resets_only_selected_fifo(self):
        bias = torch.zeros(12)
        actuator = BatchedPACEActuator(bias, 3)
        q = torch.zeros(2, 12)
        qdot = torch.zeros_like(q)
        for target_value in (1.0, 2.0, 3.0):
            actuator.step(torch.full_like(q, target_value), q, qdot)
        actuator.reset_envs(torch.tensor([0]), q)
        output = actuator.step(torch.full_like(q, 4.0), q, qdot).applied_torque
        self.assertTrue(torch.equal(output[0], torch.zeros(12)))
        self.assertTrue(torch.all(output[1] > 0.0))

    def test_reset_pose_respects_frozen_encoder_frame(self):
        q0 = torch.tensor([[0.0, 0.4, -0.8]])
        multiplier = torch.tensor([[1.0, 0.5, 1.5]])
        bias = torch.tensor([0.1, -0.2, 0.3])
        q_true = reset_true_joint_position(q0, multiplier, bias)
        torch.testing.assert_close(q_true - bias, q0 * multiplier)


class Stage1CommandResetTimingTest(unittest.TestCase):
    def test_command_sampling_is_seed_reproducible_and_deadband_is_exact(self):
        first = sample_commands(100, torch.Generator().manual_seed(9), device=torch.device("cpu"))
        second = sample_commands(100, torch.Generator().manual_seed(9), device=torch.device("cpu"))
        torch.testing.assert_close(first, second)
        norms = torch.linalg.vector_norm(first[:, :2], dim=1)
        self.assertTrue(torch.all((norms == 0) | (norms > 0.2)))

    def test_command_resampling_and_strict_timeout(self):
        lengths = torch.tensor([0, 999, 1000, 1001, 2000, 2001])
        self.assertEqual(
            command_resample_mask(lengths, 1000).tolist(),
            [True, False, True, False, True, False],
        )
        self.assertEqual(timeout_mask(lengths, 2000).tolist(), [False] * 5 + [True])
        self.assertEqual(STAGE1_CONFIG.reset.terminate_contact_bodies, ("base",))
        self.assertIn("none", STAGE1_CONFIG.reset.base_orientation_condition)
        self.assertIn("none", STAGE1_CONFIG.reset.base_height_condition)


class Stage1RewardTest(unittest.TestCase):
    def _reward(self, *, iteration=0, collision=True, touchdown=1.0):
        zeros3 = torch.zeros(1, 3)
        return compute_reward_terms(
            torch.zeros(1, 4),
            zeros3,
            zeros3,
            torch.zeros(1, 12),
            torch.zeros(1, 12),
            torch.tensor([10.0]),
            torch.zeros(1, 1, 3),
            torch.tensor([collision]),
            torch.tensor([[touchdown, 0.0, 0.0, 0.0]]),
            iteration=iteration,
        )

    def test_four_term_manifest_and_scales(self):
        manifest = reward_manifest()
        self.assertEqual([item["name"] for item in manifest], [
            "velocity_tracking", "energy", "collision", "foot_touchdown"
        ])
        self.assertEqual([item["scale"] for item in manifest], [0.2, -0.00016, -1.0, -0.1])

    def test_reward_computation_and_policy_dt_integration(self):
        terms = self._reward(iteration=0)
        self.assertAlmostEqual(terms.velocity_tracking.item(), 2.0)
        self.assertAlmostEqual(terms.energy.item(), 0.0)
        self.assertAlmostEqual(terms.collision.item(), 1.0)
        self.assertAlmostEqual(terms.foot_touchdown.item(), 1.0)
        self.assertAlmostEqual(terms.total.item(), -0.006, places=7)

    def test_penalty_timing_has_500_iteration_half_life(self):
        self.assertEqual(penalty_schedule(0), 0.0)
        self.assertAlmostEqual(penalty_schedule(500), 0.5)
        terms = self._reward(iteration=500, collision=False)
        self.assertAlmostEqual(terms.total.item(), 0.0035, places=7)


class Stage1TerrainPPOAndProvenanceTest(unittest.TestCase):
    def test_terrain_and_domain_randomization_are_frozen(self):
        terrain = STAGE1_CONFIG.terrain
        randomization = STAGE1_CONFIG.randomization
        self.assertEqual(terrain.mesh_type, "trimesh")
        self.assertTrue(terrain.curriculum)
        self.assertEqual(sum(terrain.terrain_proportions), 1.0)
        self.assertEqual(terrain.terrain_types[-1], "discrete_boxes")
        self.assertTrue(randomization.randomize_ground_friction)
        self.assertTrue(randomization.push_robots)
        self.assertFalse(randomization.randomize_dynamics)
        self.assertFalse(randomization.randomize_base_mass)
        self.assertFalse(randomization.randomize_motor_strength)

    def test_ppo_config_matches_fixed_references(self):
        train = ppo_train_cfg()
        self.assertEqual(train["policy"]["actor_hidden_dims"], [512, 256, 128])
        self.assertEqual(train["policy"]["critic_hidden_dims"], [512, 256, 128])
        self.assertEqual(train["algorithm"]["learning_rate"], 1.0e-3)
        self.assertEqual(train["algorithm"]["gamma"], 0.99)
        self.assertEqual(train["algorithm"]["lam"], 0.95)
        self.assertEqual(train["algorithm"]["clip_param"], 0.2)
        self.assertEqual(train["runner"]["num_steps_per_env"], 24)
        self.assertEqual(train["runner"]["max_iterations"], 1500)

    def test_rsl_rl_timeout_bootstrap_behavior(self):
        actual = timeout_bootstrap(
            torch.tensor([1.0, 1.0]),
            torch.tensor([[2.0], [2.0]]),
            torch.tensor([False, True]),
            0.99,
        )
        torch.testing.assert_close(actual, torch.tensor([1.0, 2.98]))

    def test_iteration_bridge_updates_penalty_clock_and_entropy(self):
        class Algorithm:
            entropy_coef = None

        class Environment:
            iteration = None

            def set_training_iteration(self, value):
                self.iteration = value

        algorithm, environment = Algorithm(), Environment()
        coefficient = apply_iteration_schedules(algorithm, environment, 750)
        self.assertEqual(environment.iteration, 750)
        self.assertAlmostEqual(coefficient, (0.01 + 0.001) / 2.0)
        self.assertEqual(algorithm.entropy_coef, coefficient)
        self.assertGreater(entropy_schedule(0, 0.01, 0.001, 750, 0.01), coefficient)

    def test_manifest_provenance_and_ppo_guard(self):
        self.assertFalse(MANIFEST["status"]["PPO_training_started"])
        self.assertEqual(MANIFEST["status"]["Stage_0C_E_exact_legacy_replay"], "FAIL / unresolved")
        self.assertFalse(MANIFEST["training_guard"]["formal_PPO_allowed_in_this_stage"])
        for item in MANIFEST["unresolved_assumptions"]:
            self.assertTrue({"value", "provenance", "source_locator", "confidence", "notes"} <= item.keys())
            self.assertNotEqual(item["provenance"], "PACE_explicit")

    def test_stage0_frozen_files_are_byte_identical(self):
        for relative, expected in MANIFEST["stage0_integrity"]["files"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_evaluation_protocol_is_frozen_before_training(self):
        evaluation = STAGE1_CONFIG.evaluation
        self.assertEqual(evaluation.seeds, (0, 1, 2))
        self.assertEqual(evaluation.checkpoint_selection, "final iteration; no best-on-evaluation selection")
        self.assertFalse(evaluation.actor_observation_noise)
        self.assertFalse(evaluation.pushes)

    def test_phase_a_authorization_is_bounded_and_non_paper(self):
        phase = STAGE1_CONFIG.phase_a_validation
        authorized = AUTHORIZATION["authorized_first_run"]
        self.assertEqual(AUTHORIZATION["status"]["Stage_1"], "IMPLEMENTED / PPO READY")
        self.assertFalse(AUTHORIZATION["status"]["PPO_started"])
        self.assertEqual(phase.num_envs, 1024)
        self.assertEqual(phase.max_iterations, 300)
        self.assertEqual(phase.seed, 1)
        self.assertEqual(phase.sim_device, "cuda:0")
        self.assertFalse(phase.formal_baseline)
        self.assertEqual(authorized["experiment_name"], phase.experiment_name)
        self.assertFalse(authorized["formal_4096_env_baseline_authorized"])

    def test_energy_interpretation_boundary_is_explicit(self):
        guard = AUTHORIZATION["interpretation_guard"]
        self.assertIn("unavailable electrical conversion parameters", guard["allowed"])
        self.assertEqual(guard["forbidden"], "exactly reproduces PACE energy model")


if __name__ == "__main__":
    unittest.main()
