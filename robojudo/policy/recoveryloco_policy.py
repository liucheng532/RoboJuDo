from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import RecoveryLocoPolicyCfg
from robojudo.utils.util_func import get_gravity_orientation

if TYPE_CHECKING:
    from robojudo.environment.utils.mujoco_viz import MujocoVisualizer

logger = logging.getLogger(__name__)


@policy_registry.register
class RecoveryLocoPolicy(Policy):
    """Unified G1 locomotion and fall-recovery policy trained in AMP_mjlab."""

    cfg_policy: RecoveryLocoPolicyCfg

    def __init__(self, cfg_policy: RecoveryLocoPolicyCfg, device: str = "cpu"):
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if device in {"cuda", "tensorrt"}:
            preferred = []
            if device == "tensorrt":
                preferred.append("TensorrtExecutionProvider")
            preferred.append("CUDAExecutionProvider")
            providers = [provider for provider in preferred if provider in available] + ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(cfg_policy.policy_file, providers=providers)
        self.input_names = [model_input.name for model_input in self.session.get_inputs()]
        self.output_names = [model_output.name for model_output in self.session.get_outputs()]

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.expected_obs_size = cfg_policy.observation_size
        self.frame_obs_size = cfg_policy.frame_observation_size
        self.expected_output_size = cfg_policy.output_size
        self.obs_clip = cfg_policy.obs_clip
        self.action_scales = np.asarray(cfg_policy.action_scales, dtype=np.float32)
        self.command_ranges = np.asarray(cfg_policy.command_ranges, dtype=np.float32)
        self.command_deadzone = cfg_policy.command_deadzone
        self.command_smoothing = cfg_policy.command_smoothing
        self.keyboard_value = cfg_policy.keyboard_value

        self._validate_model_signature()
        self.reset()

    def _validate_model_signature(self):
        if len(self.input_names) != 1 or self.input_names[0] != "obs":
            raise ValueError(f"RecoveryLocoPolicy expects one ONNX input named 'obs', got {self.input_names}")
        if len(self.output_names) < 1 or self.output_names[0] != "actions":
            raise ValueError(
                f"RecoveryLocoPolicy expects the first ONNX output to be 'actions', got {self.output_names}"
            )

        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if isinstance(input_shape[-1], int) and input_shape[-1] != self.expected_obs_size:
            raise ValueError(
                f"RecoveryLocoPolicy ONNX observation size mismatch: {input_shape[-1]} != {self.expected_obs_size}"
            )
        if isinstance(output_shape[-1], int) and output_shape[-1] != self.expected_output_size:
            raise ValueError(
                f"RecoveryLocoPolicy ONNX output size mismatch: {output_shape[-1]} != {self.expected_output_size}"
            )

    def reset(self):
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.current_commands = np.zeros(3, dtype=np.float32)
        self.command_history = np.zeros(3, dtype=np.float32)
        self.pressed_keys: set[str] = set()
        self.history_buf: deque[np.ndarray] = deque(maxlen=self.history_length)

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1

    def _map_axis(self, value: float, target_range: np.ndarray) -> float:
        if abs(value) < self.command_deadzone:
            return float(target_range[1])
        if value < 0.0:
            return float(target_range[1] + value * (target_range[1] - target_range[0]))
        return float(target_range[1] + value * (target_range[2] - target_range[1]))

    def _get_joystick_commands(self, ctrl_data) -> np.ndarray | None:
        for controller_name in ("JoystickCtrl", "UnitreeCtrl"):
            if controller_name not in ctrl_data:
                continue
            axes = ctrl_data[controller_name]["axes"]
            raw_commands = np.asarray(
                [
                    axes.get("LeftY", 0.0),
                    axes.get("LeftX", 0.0),
                    axes.get("RightX", 0.0),
                ],
                dtype=np.float32,
            )
            return np.asarray(
                [
                    self._map_axis(value, target_range)
                    for value, target_range in zip(raw_commands, self.command_ranges, strict=True)
                ],
                dtype=np.float32,
            )
        return None

    def _get_keyboard_commands(self, ctrl_data) -> np.ndarray:
        keyboard_data = ctrl_data.get("KeyboardCtrl")
        if keyboard_data is not None:
            for event in keyboard_data.get("keyboard_event", []):
                if event.get("type") != "keyboard":
                    continue
                name = event.get("name")
                if name not in {"w", "s", "a", "d", "q", "e"}:
                    continue
                if event.get("pressed"):
                    self.pressed_keys.add(name)
                else:
                    self.pressed_keys.discard(name)

        raw_commands = np.asarray(
            [
                float("w" in self.pressed_keys) - float("s" in self.pressed_keys),
                float("d" in self.pressed_keys) - float("a" in self.pressed_keys),
                float("e" in self.pressed_keys) - float("q" in self.pressed_keys),
            ],
            dtype=np.float32,
        )
        raw_commands *= self.keyboard_value
        raw_commands = np.clip(raw_commands, -1.0, 1.0)
        return np.asarray(
            [
                self._map_axis(value, target_range)
                for value, target_range in zip(raw_commands, self.command_ranges, strict=True)
            ],
            dtype=np.float32,
        )

    def _get_commands(self, ctrl_data) -> np.ndarray:
        commands = self._get_joystick_commands(ctrl_data)
        if commands is None:
            commands = self._get_keyboard_commands(ctrl_data)

        commands = self.command_history * self.command_smoothing + commands * (1.0 - self.command_smoothing)
        self.command_history = commands.astype(np.float32, copy=True)
        self.current_commands = self.command_history.copy()
        return self.current_commands

    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        projected_gravity = np.asarray(get_gravity_orientation(env_data.base_quat), dtype=np.float32)
        commands = self._get_commands(ctrl_data)
        dof_pos_rel = np.asarray(env_data.dof_pos - self.default_dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)

        current_state = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                commands,
                dof_pos_rel,
                dof_vel,
                self.last_action,
            ]
        ).astype(np.float32)
        if current_state.shape != (self.frame_obs_size,):
            raise ValueError(
                f"RecoveryLocoPolicy frame observation size mismatch: {current_state.shape[0]} != {self.frame_obs_size}"
            )

        if not self.history_buf:
            for _ in range(self.history_length):
                self.history_buf.append(current_state.copy())
        else:
            self.history_buf.append(current_state.copy())

        obs = np.concatenate(tuple(self.history_buf)).astype(np.float32)
        obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        if obs.shape != (self.expected_obs_size,):
            raise ValueError(
                f"RecoveryLocoPolicy observation size mismatch: {obs.shape[0]} != {self.expected_obs_size}"
            )

        return obs, {"commands": commands}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        inputs = {self.input_names[0]: obs.reshape(1, -1).astype(np.float32)}
        output = self.session.run(self.output_names[:1], inputs)[0]
        actions = np.asarray(output, dtype=np.float32).reshape(-1)
        if actions.shape != (self.expected_output_size,):
            raise ValueError(
                f"RecoveryLocoPolicy output size mismatch: {actions.shape[0]} != {self.expected_output_size}"
            )

        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        self.last_action = actions.copy()
        return actions * self.action_scale * self.action_scales

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = env_data.get("base_pos", None)
        if base_pos is None:
            return
        base_quat = env_data["base_quat"]
        command_x, command_y, command_yaw = extras.get("commands", np.zeros(3))
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [command_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,
        )
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, command_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,
        )
        visualizer.draw_arrow(
            base_pos + np.asarray([0.0, 0.0, 0.8]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )
