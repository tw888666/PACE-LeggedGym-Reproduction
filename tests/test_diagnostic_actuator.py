import unittest

import torch

from pace_stage0.actuator import PACEActuatorCore
from pace_stage0.constants import KP
from pace_stage0.diagnostic_actuator import (
    BiasLawDiagnosticActuator,
    LEGACY_EFFECTIVE_BIAS,
    PUBLIC_ENCODER_FEEDBACK,
)


class BiasLawDiagnosticActuatorTest(unittest.TestCase):
    def test_public_mode_is_exactly_formal_core(self):
        bias = torch.tensor([0.02, -0.01, 0.005])
        formal = PACEActuatorCore(bias, delay_steps=3)
        diagnostic = BiasLawDiagnosticActuator(
            bias, delay_steps=3, mode=PUBLIC_ENCODER_FEEDBACK
        )
        q = torch.tensor([0.1, -0.2, 0.3])
        formal.reset(q)
        diagnostic.reset(q)
        for index in range(8):
            target = q + 0.01 * (index + 1)
            qdot = torch.tensor([0.1, -0.2, 0.3]) * index
            formal_step = formal.step(target, q, qdot)
            diagnostic_step = diagnostic.step(target, q, qdot)
            torch.testing.assert_close(
                diagnostic_step.q_control, formal_step.q_encoder, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                diagnostic_step.raw_pd_torque,
                formal_step.raw_pd_torque,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                diagnostic_step.saturated_torque,
                formal_step.saturated_torque,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                diagnostic_step.applied_torque,
                formal_step.applied_torque,
                rtol=0.0,
                atol=0.0,
            )

    def test_legacy_mode_changes_only_effective_bias_term(self):
        bias = torch.tensor([0.02, -0.01, 0.005])
        public = BiasLawDiagnosticActuator(
            bias, delay_steps=0, mode=PUBLIC_ENCODER_FEEDBACK
        )
        legacy = BiasLawDiagnosticActuator(
            bias, delay_steps=0, mode=LEGACY_EFFECTIVE_BIAS
        )
        target = torch.tensor([0.2, -0.1, 0.4])
        q_true = torch.tensor([0.1, -0.2, 0.3])
        qdot = torch.tensor([0.3, -0.4, 0.2])
        public_step = public.step(target, q_true, qdot)
        legacy_step = legacy.step(target, q_true, qdot)
        torch.testing.assert_close(
            legacy_step.raw_pd_torque,
            public_step.raw_pd_torque - KP * bias,
            rtol=0.0,
            atol=2e-6,
        )
        torch.testing.assert_close(legacy_step.q_control, q_true)
        torch.testing.assert_close(
            public.comparison_position(q_true),
            legacy.comparison_position(q_true),
            rtol=0.0,
            atol=0.0,
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            BiasLawDiagnosticActuator(torch.zeros(3), 3, "auto_select")


if __name__ == "__main__":
    unittest.main()
