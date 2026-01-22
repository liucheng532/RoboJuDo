import logging

import numpy as np
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PromptMimicPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import (
    matrix_from_quat,
    subtract_frame_transforms,
    quat_rotate_inverse_np,
    get_gravity_orientation,
)

logger = logging.getLogger(__name__)

# --- Constants copied from TextOpTracker deploy_mujoco.py (training obs contract) ---
ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# default joint pos per training script
DEFAULT_JOINT_POS = np.array(
    [
        -0.312,
        -0.312,  # hip_pitch L/R
        0.0,  # waist_yaw
        0.0,
        0.0,  # hip_roll L/R
        0.0,  # waist_roll
        0.0,
        0.0,  # hip_yaw L/R
        0.0,  # waist_pitch
        0.669,
        0.669,  # knee
        0.2,
        0.2,  # shoulder_pitch
        -0.363,
        -0.363,  # ankle_pitch
        0.2,
        -0.2,  # shoulder_roll
        0.0,
        0.0,  # ankle_roll
        0.0,
        0.0,  # shoulder_yaw
        0.6,
        0.6,  # elbow
        0.0,
        0.0,  # wrist_roll
        0.0,
        0.0,  # wrist_pitch
        0.0,
        0.0,  # wrist_yaw
    ],
    dtype=np.float32,
)


@policy_registry.register
class PromptMimicPolicy(Policy):
    """
    Similar to BeyondMimicPolicy(use_motion_from_model=False):
    - ONNX produces actions only
    - Motion command comes from ctrl_data["PromptMimicCtrl"]
    """

    cfg_policy: PromptMimicPolicyCfg

    def __init__(self, cfg_policy: PromptMimicPolicyCfg, device):
        sess_options = ort.SessionOptions()

        device = "cpu"
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(f"Unknown device: {device}")

        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.motion_anchor_body_index = -1

        cfg_policy_new = cfg_policy.model_copy()
        # For PromptMimic we do NOT rely on modelmeta; use cfg-provided dof/action scales.
        super().__init__(cfg_policy=cfg_policy_new, device=device)

        self.action_scales = np.asarray(self.cfg_policy.action_scales)
        self.without_state_estimator = self.cfg_policy.without_state_estimator
        self.override_robot_anchor_pos = self.cfg_policy.override_robot_anchor_pos
        self.use_motion_from_model = False  # explicit

        self.max_timestep = self.cfg_policy.max_timestep
        self.command = None
        self.reset()

    def _prepare_policy(self):
        obs_shape = self.session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        self.get_action(obs)

    def reset(self):
        self.timestep: float = self.cfg_policy.start_timestep
        self.pbar = ProgressBar(f"PromptMimic {self.cfg_policy.policy_name}", self.max_timestep) if self.max_timestep > 0 else None
        self.play_speed: float = 1.0
        self.flag_motion_done = False
        self._prepare_policy()

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1 * self.play_speed
        if self.pbar:
            self.pbar.set(self.timestep)

        if 0 < self.max_timestep <= self.timestep:
            self.play_speed = 0.0
            self.flag_motion_done = True

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0

    def _get_command(self, env_data, ctrl_data):
        assert "PromptMimicCtrl" in ctrl_data, "PromptMimicCtrl not found in ctrl_data"
        command = ctrl_data.get("PromptMimicCtrl")
        self.command = command
        return (
            command["command"],
            command["robot_anchor_pos_w"],
            command["robot_anchor_quat_w"],
            command["anchor_pos_w"],
            command["anchor_quat_w"],
            command.get("hand_pose", None),
        )

    def get_observation(self, env_data, ctrl_data):
        # env_data has been adapted to policy joint order by PolicyWrapper.obs_adapter
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        ang_vel = env_data.base_ang_vel  # already body frame in MujocoEnv
        lin_vel = env_data.base_lin_vel  # already body frame in MujocoEnv
        projected_gravity = get_gravity_orientation(env_data.base_quat) if hasattr(env_data, "base_quat") else np.zeros(3, dtype=np.float32)

        (
            command,
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
            hand_pose,
        ) = self._get_command(env_data, ctrl_data)

        # Future slices from ctrl (precomputed), assumed in same joint order as training
        future_joint_pos = ctrl_data.get("future_joint_pos")
        future_joint_vel = ctrl_data.get("future_joint_vel")
        future_anchor_pos_w = ctrl_data.get("future_anchor_pos_w")
        future_anchor_quat_w = ctrl_data.get("future_anchor_quat_w")

        # current relative pose
        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        mat = matrix_from_quat(ori)

        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        obs_base_lin_vel = lin_vel
        obs_base_ang_vel = ang_vel

        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        obs_last_action = self.last_action

        # future components (if provided)
        future_cmd_flat = []
        future_anchor_rel_flat = []
        if future_joint_pos is not None and future_joint_vel is not None:
            future_cmd_flat = np.concatenate([future_joint_pos.reshape(-1), future_joint_vel.reshape(-1)])
        if future_anchor_pos_w is not None and future_anchor_quat_w is not None:
            # compute relative pose for each future step w.r.t current robot anchor
            for i in range(future_anchor_pos_w.shape[0]):
                f_pos, f_ori = subtract_frame_transforms(
                    robot_anchor_pos_w,
                    robot_anchor_quat_w,
                    future_anchor_pos_w[i],
                    future_anchor_quat_w[i],
                )
                f_mat = matrix_from_quat(f_ori)
                future_anchor_rel_flat.append(f_pos)
                future_anchor_rel_flat.append(f_mat[:, :2].flatten())
            if future_anchor_rel_flat:
                future_anchor_rel_flat = np.concatenate(future_anchor_rel_flat)
            else:
                future_anchor_rel_flat = []

        # GeneralMotionTracking observation order (431 dims):
        # future_cmd(290) + future_anchor_rel(45) + projected_gravity(3) + 
        # base_lin_vel(3) + base_ang_vel(3) + joint_pos_rel(29) + joint_vel_rel(29) + last_action(29)
        obs_prop = np.concatenate(
            [
                future_cmd_flat,              # 290 dims (5 steps * 29 joints * 2)
                future_anchor_rel_flat,       # 45 dims (5 steps * (3 pos + 6 ori))
                projected_gravity,            # 3 dims
                obs_base_lin_vel,             # 3 dims (body frame, always included)
                obs_base_ang_vel,             # 3 dims (body frame)
                obs_joint_pos_rel,            # 29 dims
                obs_joint_vel_rel,            # 29 dims
                obs_last_action,              # 29 dims
            ]
        )

        # Pad/trim to exact 431 dims as required by the model
        target_dim = 431
        if len(obs_prop) < target_dim:
            obs = np.concatenate([obs_prop, np.zeros(target_dim - len(obs_prop), dtype=np.float32)])
        else:
            obs = obs_prop[:target_dim]

        extras = {
            "pos": pos,
            "ori": ori,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "command": command,
            "hand_pose": hand_pose,
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Build input feed dynamically to match the model's declared inputs.
        ort_inputs = {}
        for inp in self.session.get_inputs():
            name = inp.name
            shape = inp.shape
            if name == "obs":
                ort_inputs[name] = np.expand_dims(obs, axis=0).astype(np.float32)
            elif name == "time_step":
                ort_inputs[name] = np.expand_dims(np.array([int(self.timestep)]), axis=0).astype(np.float32)
            else:
                # Fallback: feed zeros with the same batch dim=1
                # Replace any symbolic dims with 1
                dims = []
                for s in shape:
                    if isinstance(s, str) or s is None:
                        dims.append(1)
                    else:
                        dims.append(int(s))
                # ensure batch dimension = 1
                if len(dims) == 0 or dims[0] != 1:
                    dims = [1] + dims[1:]
                ort_inputs[name] = np.zeros(dims, dtype=np.float32)

        ort_outputs = self.session.run(
            [
                "actions",
            ],
            ort_inputs,
        )
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        scaled_actions = actions * self.action_scales
        return scaled_actions

    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return first frame of the reference motion from ctrl.
        """
        if self.command is not None:
            return self.command["joint_pos"].copy()
        else:
            return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        robot_anchor_pos_w = extras["robot_anchor_pos_w"]
        robot_anchor_quat_w = extras["robot_anchor_quat_w"]
        anchor_pos_w = extras["anchor_pos_w"]
        anchor_quat_w = extras["anchor_quat_w"]

        pos = extras["pos"]

        visualizer.draw_arrow(anchor_pos_w, anchor_quat_w, [0.2, 0, 0], color=[1, 0, 0, 1], scale=2, id=0)
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            [0.2, 0, 0],
            color=[0, 1, 0, 1],
            scale=2,
            id=1,
        )
        visualizer.draw_arrow(robot_anchor_pos_w, robot_anchor_quat_w, pos, color=[0, 1, 1, 1], scale=2, id=2)

