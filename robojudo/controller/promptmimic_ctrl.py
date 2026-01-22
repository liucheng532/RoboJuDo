import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import PromptMimicCtrlCfg
from robojudo.environment import Environment
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment

logger = logging.getLogger(__name__)


class MotionLoader:
    """
    Same data contract as BeyondMimicCtrl:
    npz must contain keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w
    Optional: body_lin_vel_w, body_ang_vel_w, hand_pose
    """

    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        path = Path(motion_file)
        if not path.is_file():
            raise FileNotFoundError(f"Invalid motion file: {motion_file}")
        data = np.load(path)
        for key in ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]:
            if key not in data:
                raise ValueError(f"Motion file {motion_file} missing key: {key}")

        self.fps = data["fps"]
        self.joint_pos = data["joint_pos"]
        self.joint_vel = data["joint_vel"]
        self._body_pos_w = data["body_pos_w"]
        # Normalize quaternions (matching GeneralMotionTracking)
        self._body_quat_w = self._normalize_quat_wxyz(data["body_quat_w"].astype(np.float32))
        self._body_lin_vel_w = data.get("body_lin_vel_w", None)
        self._body_ang_vel_w = data.get("body_ang_vel_w", None)
        self._hand_pose = data.get("hand_pose", None)

        self._body_indexes = list(body_indexes)
        self.time_step_total = self.joint_pos.shape[0]

    @staticmethod
    def _normalize_quat_wxyz(quat_wxyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize quaternion along the last dim. Input shape [..., 4] in wxyz format."""
        q = quat_wxyz.astype(np.float32, copy=False)
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        n = np.maximum(n, eps)
        return q / n

    @property
    def body_pos_w(self) -> np.ndarray:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> np.ndarray:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> Optional[np.ndarray]:
        if self._body_lin_vel_w is None:
            return None
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> Optional[np.ndarray]:
        if self._body_ang_vel_w is None:
            return None
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def hand_pose(self) -> Optional[np.ndarray]:
        return self._hand_pose


@ctrl_registry.register
class PromptMimicCtrl(Controller):
    """
    Loads motion commands from npz files (multiple candidates) and allows switching among them.
    Motion data format matches BeyondMimicCtrl expectations.
    """

    cfg_ctrl: PromptMimicCtrlCfg
    env: Environment

    def __init__(self, cfg_ctrl: PromptMimicCtrlCfg, env, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        assert self.env is not None, "Env is required for PromptMimicCtrl"

        self.motion_names = list(self.cfg_ctrl.motion_names)
        if not self.motion_names:
            raise ValueError("PromptMimicCtrl requires at least one motion name")
        self.motion_idx = min(max(self.cfg_ctrl.motion_idx, 0), len(self.motion_names) - 1)

        self.override_robot_anchor_pos = self.cfg_ctrl.override_robot_anchor_pos

        # precompute name->index for all bodies
        cfg = self.cfg_ctrl.motion_cfg
        self.body_indexes = [cfg.body_names_all.index(name) for name in cfg.body_names]
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)

        self.motion: MotionLoader | None = None
        self.timestep = 0
        self.playing = False
        self.motion_init_align = TransformAlignment(yaw_only=True, xy_only=True)

        self._load_motion_by_idx(self.motion_idx)
        self.reset()

    # --- internal helpers ---
    def _load_motion(self, motion_name: str):
        motion_file = self.cfg_ctrl.motion_path(motion_name)
        self.motion = MotionLoader(motion_file, self.body_indexes, device="cpu")
        logger.info(f"[PromptMimicCtrl] Loaded motion: {motion_name} ({motion_file}) "
                    f"len={self.motion.time_step_total}, fps={self.motion.fps}")

    def _load_motion_by_idx(self, idx: int):
        self.motion_idx = idx % len(self.motion_names)
        motion_name = self.motion_names[self.motion_idx]
        self._load_motion(motion_name)
        self.reset()

    # --- public controls for switching ---
    def toggle_next_motion(self):
        self._load_motion_by_idx(self.motion_idx + 1)

    def toggle_prev_motion(self):
        self._load_motion_by_idx(self.motion_idx - 1)

    # --- properties used by policy ---
    @property
    def command(self) -> np.ndarray:
        return np.concatenate([self.joint_pos, self.joint_vel], axis=-1)

    @property
    def joint_pos(self) -> np.ndarray:
        assert self.motion is not None
        return self.motion.joint_pos[self.timestep].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        assert self.motion is not None
        return self.motion.joint_vel[self.timestep].copy()

    @property
    def anchor_pos_w(self) -> np.ndarray:
        assert self.motion is not None
        anchor_pos_w_raw = self.motion.body_pos_w[self.timestep, self.motion_anchor_body_index].copy()
        anchor_pos_w = self.motion_init_align.align_pos(anchor_pos_w_raw)
        return anchor_pos_w

    @property
    def anchor_quat_w(self) -> np.ndarray:
        assert self.motion is not None
        anchor_quat_w_raw = self.motion.body_quat_w[self.timestep, self.motion_anchor_body_index].copy()[[1, 2, 3, 0]]
        return self.motion_init_align.align_quat(anchor_quat_w_raw)

    @property
    def robot_anchor_pos_w(self) -> np.ndarray:
        if self.override_robot_anchor_pos:
            return self.anchor_pos_w
        # Use pelvis as anchor (matching GeneralMotionTracking)
        anchor_name = self.cfg_ctrl.motion_cfg.anchor_body_name
        fk_info = self.env.fk_info
        if fk_info is not None and anchor_name in fk_info:
            return fk_info[anchor_name]["pos"].copy()
        # Fallback to torso if pelvis not available
        base_pos = self.env.torso_pos
        assert base_pos is not None
        return base_pos

    @property
    def robot_anchor_quat_w(self) -> np.ndarray:
        # Use pelvis as anchor (matching GeneralMotionTracking)
        anchor_name = self.cfg_ctrl.motion_cfg.anchor_body_name
        fk_info = self.env.fk_info
        if fk_info is not None and anchor_name in fk_info:
            return fk_info[anchor_name]["quat"].copy()
        # Fallback to torso if pelvis not available
        torso_quat = self.env.torso_quat
        assert torso_quat is not None
        return torso_quat

    @property
    def hand_pose(self) -> np.ndarray | None:
        assert self.motion is not None
        hand_pose = self.motion.hand_pose
        if hand_pose is not None:
            hand_pose = hand_pose[self.timestep].copy()
            if len(hand_pose.shape) == 1:
                hand_dim = hand_pose.shape[0] // 2
                hand_pose = hand_pose.reshape(2, hand_dim)
            return hand_pose
        return None

    # --- lifecycle ---
    def reset(self):
        assert self.motion is not None
        self.timestep = 0
        self.playing = False
        self.pbar = ProgressBar(
            f"PromptMimicCtrl {self.motion_names[self.motion_idx]}",
            self.motion.time_step_total,
        )

        # align the robot to the motion's starting pose
        init2anchor_pos = self.motion.body_pos_w[0, self.motion_anchor_body_index].copy()
        init2anchor_quat = self.motion.body_quat_w[0, self.motion_anchor_body_index].copy()[[1, 2, 3, 0]]
        self.motion_init_align.set_base(quat=init2anchor_quat, pos=init2anchor_pos)

    def post_step_callback(self, commands: list[str] | None = None):
        assert self.motion is not None
        self.pbar.set(self.timestep)
        if self.timestep < self.motion.time_step_total - 1 and self.playing:
            self.timestep += 1

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.playing = True
                case "[MOTION_FADE_OUT]":
                    self.playing = False

    def get_data(self):
        assert self.motion is not None
        # Precompute future slices for policy (future_steps=5)
        future_steps = 5
        t = self.timestep
        t_indices = [min(t + i + 1, self.motion.time_step_total - 1) for i in range(future_steps)]
        future_joint_pos = self.motion.joint_pos[t_indices]
        future_joint_vel = self.motion.joint_vel[t_indices]
        future_anchor_pos_w_raw = self.motion.body_pos_w[t_indices, self.motion_anchor_body_index].copy()
        future_anchor_quat_w_raw = self.motion.body_quat_w[t_indices, self.motion_anchor_body_index].copy()[:, [1, 2, 3, 0]]

        ctrl_data = {
            "command": self.command,
            "joint_pos": self.joint_pos,
            "robot_anchor_pos_w": self.robot_anchor_pos_w,
            "robot_anchor_quat_w": self.robot_anchor_quat_w,
            "anchor_pos_w": self.anchor_pos_w,
            "anchor_quat_w": self.anchor_quat_w,
            "timestep": self.timestep,
            "hand_pose": self.hand_pose,
            "future_joint_pos": future_joint_pos,
            "future_joint_vel": future_joint_vel,
            "future_anchor_pos_w": future_anchor_pos_w_raw,
            "future_anchor_quat_w": future_anchor_quat_w_raw,
        }
        return ctrl_data


if __name__ == "__main__":
    # Simple sanity dry-run (requires a MuJoCo env to be meaningful; here just load motion)
    from robojudo.config.g1.ctrl.g1_promptmimic_ctrl_cfg import G1PromptMimicCtrlCfg
    from robojudo.environment.dummy_env import DummyEnv
    from robojudo.config.g1.env.g1_dummy_env_cfg import G1DummyEnvCfg

    env = DummyEnv(cfg_env=G1DummyEnvCfg())
    ctrl = PromptMimicCtrl(cfg_ctrl=G1PromptMimicCtrlCfg(), env=env)
    print(ctrl.get_data().keys())

