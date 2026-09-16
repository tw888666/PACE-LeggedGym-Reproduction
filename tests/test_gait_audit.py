import unittest
import numpy as np
from pace_stage1.gait_audit import selected_cases, summarize, foot_sphere_offsets


class GaitAuditTest(unittest.TestCase):
    def test_foot_offset_contains_fixed_frame_rotation(self):
        offset=foot_sphere_offsets(['LF_SHANK'])[0]
        self.assertTrue(np.allclose(offset,[.02225-.008141153063699788,-.1,-.39246+.03038319686592264]))

    def test_fixed_sample_metadata(self):
        cases=selected_cases()
        self.assertEqual(len(cases),6)
        self.assertEqual(len({c['terrain'] for c in cases}),6)
        self.assertTrue(all(c['seed']=='200000' and c['replicate']=='0' for c in cases))

    def test_periodic_contact_and_conditional_slip(self):
        contact=np.tile(np.repeat([False,True],10),5)
        arrays=dict(contact=np.tile(contact[:,None],(1,4)),
                    foot_velocity=np.zeros((100,4,3)),
                    qdot_squared_substep_mean=np.full((100,12),9.),
                    raw_action=np.full((100,12),200.),saturation=np.ones((100,12)))
        arrays['foot_velocity'][contact,:,0]=.4
        arrays['foot_velocity'][~contact,:,0]=100.
        result=summarize(arrays)
        self.assertEqual(result['joint_velocity_rms_all_rad_s'],3.)
        self.assertEqual(result['clipped_offset_abs_mean_rad'],50.)
        leg=result['legs'][0]
        self.assertEqual(leg['duty_factor'],.5)
        self.assertAlmostEqual(leg['cycle_frequency_hz'],5.)
        self.assertAlmostEqual(leg['contact_horizontal_speed_mean_m_s'],.4)
        self.assertAlmostEqual(leg['complete_stance_mean_s'],.1)

    def test_no_complete_cycle_is_missing(self):
        arrays=dict(contact=np.zeros((10,4)),foot_velocity=np.zeros((10,4,3)),
                    qdot_squared_substep_mean=np.zeros((10,12)),raw_action=np.zeros((10,12)),saturation=np.zeros((10,12)))
        result=summarize(arrays)
        self.assertIsNone(result['legs'][0]['cycle_frequency_hz'])
        self.assertIsNone(result['legs'][0]['contact_horizontal_speed_mean_m_s'])


if __name__=='__main__':unittest.main()
