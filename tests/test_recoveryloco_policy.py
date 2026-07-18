import unittest
from types import SimpleNamespace

import numpy as np

from robojudo.config.config_manager import ConfigManager
from robojudo.config.g1.policy.g1_locomode_policy_cfg import G1LocoModePolicyCfg
from robojudo.config.g1.policy.g1_recoveryloco_policy_cfg import G1RecoveryLocoPolicyCfg
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.policy.recoveryloco_policy import RecoveryLocoPolicy


class TestRecoveryLocoPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = G1RecoveryLocoPolicyCfg()
        cls.policy = RecoveryLocoPolicy(cls.cfg, device="cpu")

    def setUp(self):
        self.policy.reset()
        self.default_pos = np.asarray(self.cfg.obs_dof.default_pos, dtype=np.float32)
        self.env_data = SimpleNamespace(
            dof_pos=self.default_pos.copy(),
            dof_vel=np.zeros(29, dtype=np.float32),
            base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
        )

    def test_new_pipeline_config_replaces_locomode_only(self):
        original_cfg = ConfigManager("g1_locomode_beyondmimic").get_cfg()
        recovery_cfg = ConfigManager("g1_recoveryloco_beyondmimic").get_cfg()

        self.assertIsInstance(original_cfg.mimic_policies[0], G1LocoModePolicyCfg)
        self.assertIsInstance(recovery_cfg.mimic_policies[0], G1RecoveryLocoPolicyCfg)
        self.assertEqual(recovery_cfg.pipeline_type, "RlLocoMimicPipeline")
        self.assertEqual(type(recovery_cfg.loco_policy), type(original_cfg.loco_policy))
        self.assertEqual(recovery_cfg.warmup_mimic_idx, 0)

    def test_real_pipeline_uses_unitree_environment_and_allows_recovery(self):
        cfg = ConfigManager("g1_recoveryloco_beyondmimic_real").get_cfg()

        self.assertEqual(cfg.env.env_type, "UnitreeCppEnv")
        self.assertEqual(cfg.env.unitree.net_if, "eth0")
        self.assertEqual(cfg.pipeline_type, "RlLocoMimicPipeline")
        self.assertIsInstance(cfg.ctrl[0], UnitreeCtrlCfg)
        self.assertEqual(cfg.ctrl[0].triggers["A"], "[SHUTDOWN]")
        self.assertIsInstance(cfg.mimic_policies[0], G1RecoveryLocoPolicyCfg)
        self.assertFalse(cfg.do_safety_check)

    def test_observation_layout_and_time_history(self):
        offsets = np.arange(29, dtype=np.float32) / 100.0
        velocities = -offsets
        self.env_data.dof_pos += offsets
        self.env_data.dof_vel = velocities
        self.env_data.base_ang_vel = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        ctrl_data = {
            "KeyboardCtrl": {"keyboard_event": []},
            "JoystickCtrl": {"axes": {"LeftX": -0.5, "LeftY": 0.5, "RightX": 0.25, "RightY": 0.0}},
        }

        obs, extras = self.policy.get_observation(self.env_data, ctrl_data)
        frame = obs[:96]

        self.assertEqual(obs.shape, (384,))
        np.testing.assert_allclose(frame[:3], self.env_data.base_ang_vel)
        np.testing.assert_allclose(frame[3:6], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(frame[6:9], [0.5, -0.5, 0.785])
        np.testing.assert_allclose(frame[9:38], offsets, atol=1e-7)
        np.testing.assert_allclose(frame[38:67], velocities)
        np.testing.assert_allclose(frame[67:96], np.zeros(29))
        np.testing.assert_allclose(extras["commands"], frame[6:9])
        for history_idx in range(1, 4):
            np.testing.assert_array_equal(frame, obs[history_idx * 96 : (history_idx + 1) * 96])

        self.env_data.base_ang_vel = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        next_obs, _ = self.policy.get_observation(self.env_data, ctrl_data)
        np.testing.assert_array_equal(next_obs[: 3 * 96], obs[96:])
        np.testing.assert_allclose(next_obs[-96:-93], [1.0, 2.0, 3.0])

    def test_onnx_inference_matches_configured_action_scaling(self):
        ctrl_data = {"JoystickCtrl": {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0}}}
        obs, _ = self.policy.get_observation(self.env_data, ctrl_data)
        raw_actions = self.policy.session.run(
            self.policy.output_names[:1],
            {self.policy.input_names[0]: obs.reshape(1, -1).astype(np.float32)},
        )[0].reshape(-1)
        expected = np.clip(raw_actions, -self.cfg.action_clip, self.cfg.action_clip)
        expected *= np.asarray(self.cfg.action_scales, dtype=np.float32)

        actions = self.policy.get_action(obs)

        self.assertEqual(actions.shape, (29,))
        self.assertTrue(np.isfinite(actions).all())
        np.testing.assert_allclose(actions, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(self.policy.last_action, raw_actions, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
