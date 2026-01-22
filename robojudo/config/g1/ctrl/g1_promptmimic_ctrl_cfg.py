from robojudo.controller.ctrl_cfgs import PromptMimicCtrlCfg


class G1PromptMimicCtrlCfg(PromptMimicCtrlCfg):
    # ==== motion config ====
    robot: str = "g1"
    motion_name: str = "robot_demo1"  # placeholder for base class requirement; actual loading uses motion_names/motion_idx
    motion_names: list[str] = [
        "robot_demo1",
        "robot_demo2",
        "robot_demo3",
        "robot_demo4",
    ]
    motion_idx: int = 0

    motion_cfg: PromptMimicCtrlCfg.MotionCommandCfg = PromptMimicCtrlCfg.MotionCommandCfg(
        anchor_body_name="torso_link",
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        body_names_all=[
            "pelvis",
            "left_hip_pitch_link",
            "right_hip_pitch_link",
            "waist_yaw_link",
            "left_hip_roll_link",
            "right_hip_roll_link",
            "waist_roll_link",
            "left_hip_yaw_link",
            "right_hip_yaw_link",
            "torso_link",
            "left_knee_link",
            "right_knee_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
            "left_wrist_pitch_link",
            "right_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
    )

