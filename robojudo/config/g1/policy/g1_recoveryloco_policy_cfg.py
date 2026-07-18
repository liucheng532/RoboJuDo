from robojudo.policy.policy_cfgs import RecoveryLocoPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

RECOVERYLOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RECOVERYLOCO_DEFAULT_POS = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
]

# Values reproduced from AMP_mjlab/deployment/include/common/mathTypes.h.
KP_5020 = 14.25062309787429
KP_7520_14 = 40.179238471373175
KP_7520_22 = 99.09842777666113
KP_5010_16 = 8.611032447370201

KD_5020 = 0.907222843292423
KD_7520_14 = 2.5578897650279457
KD_7520_22 = 6.3088018534966395
KD_5010_16 = 0.548195351665136

RECOVERYLOCO_KPS = [
    *[KP_7520_22, KP_7520_22, KP_7520_14, KP_7520_22, 2.0 * KP_5020, 2.0 * KP_5020],
    *[KP_7520_22, KP_7520_22, KP_7520_14, KP_7520_22, 2.0 * KP_5020, 2.0 * KP_5020],
    *[KP_7520_14, 2.0 * KP_5020, 2.0 * KP_5020],
    *[KP_5020, KP_5020, KP_5020, KP_5020, KP_5020, KP_5010_16, KP_5010_16],
    *[KP_5020, KP_5020, KP_5020, KP_5020, KP_5020, KP_5010_16, KP_5010_16],
]

RECOVERYLOCO_KDS = [
    *[KD_7520_22, KD_7520_22, KD_7520_14, KD_7520_22, 2.0 * KD_5020, 2.0 * KD_5020],
    *[KD_7520_22, KD_7520_22, KD_7520_14, KD_7520_22, 2.0 * KD_5020, 2.0 * KD_5020],
    *[KD_7520_14, 2.0 * KD_5020, 2.0 * KD_5020],
    *[KD_5020, KD_5020, KD_5020, KD_5020, KD_5020, KD_5010_16, KD_5010_16],
    *[KD_5020, KD_5020, KD_5020, KD_5020, KD_5020, KD_5010_16, KD_5010_16],
]

RECOVERYLOCO_TORQUE_LIMITS = [
    *[139.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[139.0, 139.0, 88.0, 139.0, 50.0, 50.0],
    *[88.0, 50.0, 50.0],
    *[25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0],
    *[25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0],
]

RECOVERYLOCO_ACTION_SCALES = [
    0.25 * torque_limit / stiffness
    for torque_limit, stiffness in zip(RECOVERYLOCO_TORQUE_LIMITS, RECOVERYLOCO_KPS, strict=True)
]


class G1RecoveryLocoDoF(DoFConfig):
    joint_names: list[str] = RECOVERYLOCO_JOINT_NAMES
    default_pos: list[float] | None = RECOVERYLOCO_DEFAULT_POS
    stiffness: list[float] | None = RECOVERYLOCO_KPS
    damping: list[float] | None = RECOVERYLOCO_KDS
    torque_limits: list[float] | None = RECOVERYLOCO_TORQUE_LIMITS


class G1RecoveryLocoPolicyCfg(RecoveryLocoPolicyCfg):
    robot: str = "g1"

    obs_dof: DoFConfig = G1RecoveryLocoDoF()
    action_dof: DoFConfig = obs_dof

    action_scales: list[float] = RECOVERYLOCO_ACTION_SCALES
