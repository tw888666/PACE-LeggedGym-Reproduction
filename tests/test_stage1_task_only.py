import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from pace_stage1.config import STAGE1_CONFIG, ppo_train_cfg
from pace_stage1.runner_bridge import apply_task_iteration
from pace_stage1.semantics import (
    BatchedPACEActuator,
    build_actor_observation,
    build_critic_observation,
    command_resample_mask,
    command_yaw_rate,
    compute_actuator_logging_metrics,
    compute_policy_target_pipeline,
    compute_task_reward_terms,
    foot_touchdown_schedule,
    make_target_adapter,
    reset_true_joint_position,
    reward_manifest,
    sample_commands,
    summarize_completed_episodes,
    timeout_bootstrap,
    timeout_mask,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "provenance" / "stage1_task_only_manifest.json").read_text(encoding="utf-8")
)


class ObservationTest(unittest.TestCase):
    @staticmethod
    def components(count=2):
        return (
            torch.ones(count, 3),
            torch.ones(count, 3) * 2,
            torch.ones(count, 3) * 3,
            torch.ones(count, 4) * 4,
            torch.ones(count, 12) * 5,
            torch.ones(count, 12) * 6,
            torch.ones(count, 12) * 7,
        )

    def test_actor_shape_order_and_scale(self):
        obs = build_actor_observation(*self.components(), add_noise=False)
        self.assertEqual(tuple(obs.shape), (2, 48))
        torch.testing.assert_close(obs[0, 0:3], torch.full((3,), 2.0))
        torch.testing.assert_close(obs[0, 3:6], torch.full((3,), 0.5))
        torch.testing.assert_close(obs[0, 9:12], torch.tensor([8.0, 8.0, 1.0]))
        torch.testing.assert_close(obs[0, 12:24], torch.full((12,), 5.0))
        torch.testing.assert_close(obs[0, 24:36], torch.full((12,), 0.3))
        torch.testing.assert_close(obs[0, 36:48], torch.full((12,), 7.0))
        self.assertEqual(STAGE1_CONFIG.observation.order[-1], "previous_policy_action[12]")

    def test_noise_and_normalization_semantics(self):
        generator = torch.Generator().manual_seed(10)
        clean = build_actor_observation(*self.components(), add_noise=False)
        noisy = build_actor_observation(*self.components(), add_noise=True, generator=generator)
        torch.testing.assert_close(clean[:, 9:12], noisy[:, 9:12])
        torch.testing.assert_close(clean[:, 36:48], noisy[:, 36:48])
        self.assertGreater(torch.max(torch.abs(clean[:, :9] - noisy[:, :9])).item(), 0.0)
        self.assertIn("no empirical", STAGE1_CONFIG.observation.normalization)

    def test_critic_shape_and_order(self):
        critic = build_critic_observation(
            torch.zeros(2, 48),
            torch.ones(2, 3),
            torch.ones(2, 3) * 2,
            torch.tensor([0.5, 0.7]),
            torch.tensor([[True, False, True, False], [False] * 4]),
            torch.zeros(2, 294),
        )
        self.assertEqual(tuple(critic.shape), (2, 353))
        torch.testing.assert_close(critic[:, 48:51], torch.ones(2, 3))
        torch.testing.assert_close(critic[:, 54], torch.tensor([0.5, 0.7]))


class ActionAndStage0InterfaceTest(unittest.TestCase):
    def test_action_mapping_order_scale_and_pose(self):
        cfg = STAGE1_CONFIG.action
        self.assertEqual(cfg.joint_order[:6], (
            "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE"
        ))
        action = torch.zeros(1, 12)
        action[0, 0] = 1.0
        pose = torch.tensor(cfg.default_joint_pose_rad).reshape(1, 12)
        target = make_target_adapter()(
            action, pose, torch.full_like(pose, -2.0), torch.full_like(pose, 2.0)
        )
        self.assertAlmostEqual(target[0, 0].item(), 0.5)
        torch.testing.assert_close(target[0, 1:], pose[0, 1:])

    def test_policy_and_physics_timing(self):
        self.assertEqual(STAGE1_CONFIG.action.physics_dt_s, 0.0025)
        self.assertEqual(STAGE1_CONFIG.action.policy_decimation, 4)
        self.assertEqual(STAGE1_CONFIG.action.policy_dt_s, 0.01)

    def test_policy_target_wrapper_can_reproduce_legacy_clamp_candidate(self):
        config = replace(
            STAGE1_CONFIG,
            target_adapter=replace(
                STAGE1_CONFIG.target_adapter,
                enforce_joint_limit_on_policy_target=True,
            ),
        )
        action = torch.tensor([[0.0, 4.0]])
        pose = torch.zeros_like(action)
        result = compute_policy_target_pipeline(
            action,
            pose,
            torch.full_like(action, -1.0),
            torch.full_like(action, 1.0),
            config,
        )
        self.assertEqual(result.saturation_mask.tolist(), [[False, True]])
        torch.testing.assert_close(result.selected_target, result.after_joint_limit)

    def test_default_policy_target_bypasses_reconstruction_clamp(self):
        self.assertFalse(
            STAGE1_CONFIG.target_adapter.enforce_joint_limit_on_policy_target
        )
        action = torch.tensor([[0.0, 4.0]])
        pose = torch.zeros_like(action)
        result = compute_policy_target_pipeline(
            action,
            pose,
            torch.full_like(action, -1.0),
            torch.full_like(action, 1.0),
        )
        torch.testing.assert_close(result.selected_target, result.before_joint_limit)
        self.assertTrue(result.saturation_mask[0, 1])

    def test_batched_actuator_changes_only_fifo_reset(self):
        actuator = BatchedPACEActuator(torch.zeros(12), 3)
        q = torch.zeros(2, 12)
        for value in (1.0, 2.0, 3.0):
            actuator.step(torch.full_like(q, value), q, q)
        actuator.reset_envs(torch.tensor([0]), q)
        output = actuator.step(torch.full_like(q, 4.0), q, q).applied_torque
        self.assertTrue(torch.equal(output[0], torch.zeros(12)))
        self.assertTrue(torch.all(output[1] > 0.0))

    def test_reset_pose_uses_frozen_encoder_frame(self):
        q0 = torch.tensor([[0.0, 0.4, -0.8]])
        multiplier = torch.tensor([[1.0, 0.5, 1.5]])
        bias = torch.tensor([0.1, -0.2, 0.3])
        true_position = reset_true_joint_position(q0, multiplier, bias)
        torch.testing.assert_close(true_position - bias, q0 * multiplier)


class CommandTest(unittest.TestCase):
    def test_sampling_ranges_deadband_and_seed(self):
        first = sample_commands(256, torch.Generator().manual_seed(9), device=torch.device("cpu"))
        second = sample_commands(256, torch.Generator().manual_seed(9), device=torch.device("cpu"))
        torch.testing.assert_close(first, second)
        norms = torch.linalg.vector_norm(first[:, :2], dim=1)
        self.assertTrue(torch.all((norms == 0) | (norms > 0.2)))
        self.assertTrue(torch.all(first[:, :3].abs() <= 1.0))

    def test_heading_law(self):
        commands = torch.tensor([[0.0, 0.0, 0.0, 3.0]])
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        actual = command_yaw_rate(commands, identity)
        self.assertEqual(actual[0, 2].item(), 1.0)

    def test_resampling_and_timeout_boundaries(self):
        lengths = torch.tensor([0, 999, 1000, 1001, 2000, 2001])
        self.assertEqual(command_resample_mask(lengths, 1000).tolist(), [True, False, True, False, True, False])
        self.assertEqual(timeout_mask(lengths, 2000).tolist(), [False] * 5 + [True])


class TaskRewardExecutionTest(unittest.TestCase):
    @staticmethod
    def reward(*, collision=False, touchdown=0.0, terminated=False, timeout=False, iteration=0, config=STAGE1_CONFIG):
        zeros3 = torch.zeros(1, 3)
        return compute_task_reward_terms(
            torch.zeros(1, 4),
            zeros3,
            zeros3,
            torch.tensor([float(collision)]),
            torch.tensor([[touchdown, 0.0, 0.0, 0.0]]),
            torch.tensor([terminated]),
            torch.tensor([timeout]),
            iteration=iteration,
            config=config,
        )

    def test_energy_off_reward_manifest(self):
        names = [term["name"] for term in reward_manifest()]
        self.assertEqual(names, ["velocity_tracking", "collision", "foot_touchdown", "termination"])
        self.assertFalse(any("energy" in name for name in names))

    def test_reward_dt_and_aggregation(self):
        terms = self.reward()
        self.assertAlmostEqual(terms.velocity_tracking.item(), 2.0)
        self.assertAlmostEqual(terms.scaled_velocity_tracking.item(), 0.004)
        self.assertAlmostEqual(terms.total.item(), 0.004)

    def test_only_positive_rewards_clips_negative_aggregate(self):
        terms = self.reward(collision=True)
        self.assertAlmostEqual(terms.task_before_clip.item(), -0.006, places=7)
        self.assertEqual(terms.task_after_clip.item(), 0.0)
        self.assertEqual(terms.total.item(), 0.0)

    def test_only_positive_rewards_can_be_disabled_explicitly(self):
        config = replace(
            STAGE1_CONFIG,
            rewards=replace(STAGE1_CONFIG.rewards, only_positive_rewards=False),
        )
        terms = self.reward(collision=True, config=config)
        self.assertAlmostEqual(terms.total.item(), -0.006, places=7)

    def test_termination_is_added_after_clipping_and_excludes_timeout(self):
        config = replace(
            STAGE1_CONFIG,
            rewards=replace(STAGE1_CONFIG.rewards, termination_scale=-2.0),
        )
        contact = self.reward(terminated=True, config=config)
        timeout = self.reward(terminated=True, timeout=True, config=config)
        self.assertAlmostEqual(contact.scaled_termination.item(), -0.02)
        self.assertAlmostEqual(contact.total.item(), -0.016)
        self.assertEqual(timeout.scaled_termination.item(), 0.0)
        self.assertAlmostEqual(timeout.total.item(), 0.004)

    def test_ftd_schedule_is_task_regularizer_only(self):
        self.assertEqual(foot_touchdown_schedule(0), 0.0)
        self.assertAlmostEqual(foot_touchdown_schedule(500), 0.5)
        terms = self.reward(touchdown=1.0, iteration=500)
        self.assertAlmostEqual(terms.scaled_foot_touchdown.item(), -0.0005)
        self.assertAlmostEqual(terms.total.item(), 0.0035)


class TerminationTerrainPPOProvenanceTest(unittest.TestCase):
    def test_completed_episode_logging_metrics_are_explicit(self):
        metrics = summarize_completed_episodes(
            torch.tensor([4.0, 36.0]),
            torch.tensor([1.0, 9.0]),
            torch.tensor([2.0, 9.0]),
            torch.tensor([1.0, 4.5]),
            torch.tensor([2.0, 13.5]),
            torch.tensor([4, 9]),
            torch.tensor([False, True]),
        )
        self.assertEqual(set(metrics), {
            "base_contact_rate", "timeout_rate", "velocity_tracking_rmse",
            "yaw_tracking_rmse", "mean_abs_action", "torque_saturation_ratio",
            "mean_torque_utilization",
        })
        self.assertAlmostEqual(metrics["base_contact_rate"].item(), 0.5)
        self.assertAlmostEqual(metrics["timeout_rate"].item(), 0.5)
        self.assertAlmostEqual(metrics["velocity_tracking_rmse"].item(), 1.5)
        self.assertAlmostEqual(metrics["yaw_tracking_rmse"].item(), 0.75)
        self.assertAlmostEqual(metrics["mean_abs_action"].item(), 0.75)
        self.assertAlmostEqual(metrics["torque_saturation_ratio"].item(), 0.375)
        self.assertAlmostEqual(metrics["mean_torque_utilization"].item(), 1.0)

    def test_actuator_logging_metrics_are_commanded_vs_saturated(self):
        commanded = torch.tensor([[0.0, 100.0], [89.0, -178.0]])
        saturated = torch.clamp(commanded, -89.0, 89.0)
        metrics = compute_actuator_logging_metrics(
            commanded,
            saturated,
            effort_limit_nm=89.0,
            saturation_epsilon_nm=1.0e-6,
        )
        torch.testing.assert_close(
            metrics["torque_saturation_ratio"], torch.tensor([0.5, 0.5])
        )
        torch.testing.assert_close(
            metrics["mean_torque_utilization"],
            torch.tensor([50.0 / 89.0, 1.5]),
        )

    def test_timeout_bootstrap_matches_rsl_rl(self):
        actual = timeout_bootstrap(
            torch.tensor([1.0, 1.0]),
            torch.tensor([[2.0], [2.0]]),
            torch.tensor([False, True]),
            0.99,
        )
        torch.testing.assert_close(actual, torch.tensor([1.0, 2.98]))

    def test_termination_reward_relation_is_explicit(self):
        reset = STAGE1_CONFIG.reset
        self.assertEqual(reset.terminate_contact_bodies, ("base",))
        self.assertTrue(reset.reward_before_reset)
        self.assertTrue(reset.timeout_bootstrap)
        self.assertEqual(STAGE1_CONFIG.rewards.aggregation_order[-1], "add non-timeout termination reward after clipping")

    def test_terrain_initialization_and_randomization(self):
        terrain = STAGE1_CONFIG.terrain
        randomization = STAGE1_CONFIG.randomization
        self.assertEqual(terrain.mesh_type, "trimesh")
        self.assertEqual(sum(terrain.terrain_proportions), 1.0)
        self.assertEqual((terrain.num_rows, terrain.num_cols), (10, 20))
        self.assertTrue(randomization.randomize_ground_friction)
        self.assertTrue(randomization.push_robots)
        self.assertFalse(randomization.randomize_dynamics)
        self.assertFalse(randomization.randomize_base_mass)
        self.assertFalse(randomization.randomize_motor_strength)

    def test_ppo_is_fixed_validation_tool(self):
        train = ppo_train_cfg()
        self.assertEqual(train["policy"]["actor_hidden_dims"], [512, 256, 128])
        self.assertEqual(train["policy"]["critic_hidden_dims"], [512, 256, 128])
        self.assertEqual(train["algorithm"]["gamma"], 0.99)
        self.assertEqual(train["algorithm"]["lam"], 0.95)
        self.assertEqual(train["runner"]["num_steps_per_env"], 24)
        self.assertNotIn("stage1_iteration_schedules", train)

    def test_ppo_smoke_is_bounded_and_non_experimental(self):
        smoke = STAGE1_CONFIG.ppo_smoke
        self.assertEqual(smoke.classification, "SMOKE / NON-EXPERIMENTAL")
        self.assertEqual((smoke.num_envs, smoke.rollout_steps), (2, 2))
        self.assertFalse(smoke.formal_seed)
        self.assertFalse(smoke.checkpoint_created)

    def test_formal_flat_seed0_energy_off_ablation_is_exact(self):
        ablation = STAGE1_CONFIG.formal_ablation
        self.assertEqual(ablation.experiment_name, "stage1_pace_energy_off_flat_seed0")
        self.assertEqual((ablation.num_envs, ablation.max_iterations), (4096, 3000))
        self.assertEqual(ablation.seed, 0)
        self.assertEqual(ablation.terrain_mode, "plane")
        self.assertTrue(ablation.friction_randomization)
        self.assertTrue(ablation.pushes)

    def test_runner_bridge_changes_only_task_iteration(self):
        class Environment:
            iteration = None

            def set_training_iteration(self, value):
                self.iteration = value

        environment = Environment()
        factor = apply_task_iteration(environment, 500)
        self.assertEqual(environment.iteration, 500)
        self.assertAlmostEqual(factor, 0.5)

    def test_manifest_has_required_provenance_fields(self):
        self.assertEqual(MANIFEST["history"]["start_commit"], "0fe25899e63a7be0a5c73a0184f7390352d9f6be")
        self.assertTrue(MANIFEST["status"]["PPO_validation_started"])
        self.assertFalse(MANIFEST["status"]["official_PACE_task_only_baseline_exists"])
        self.assertIn("energy reward", MANIFEST["reward"]["forbidden"])
        for item in MANIFEST["unresolved_assumptions"]:
            self.assertTrue({"value", "source", "confidence", "notes"} <= item.keys())

    def test_stage0_frozen_files_are_byte_identical(self):
        for relative, expected in MANIFEST["stage0_integrity"]["files"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_energy_off_source_has_no_prohibited_implementation(self):
        config_source = (ROOT / "src" / "pace_stage1" / "config.py").read_text(encoding="utf-8")
        semantics_source = (ROOT / "src" / "pace_stage1" / "semantics.py").read_text(encoding="utf-8")
        for prohibited in ("energy_scale", "electrical_loss", "lagrangian", "cost_critic"):
            self.assertNotIn(prohibited, config_source.lower())
            self.assertNotIn(prohibited, semantics_source.lower())


if __name__ == "__main__":
    unittest.main()
