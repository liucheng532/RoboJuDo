from robojudo.config.g1.policy.g1_beyondmimic_policy_cfg import G1BeyondMimicDoF
from robojudo.policy.policy_cfgs import PromptMimicPolicyCfg


class G1PromptMimicPolicyCfg(PromptMimicPolicyCfg):
    robot: str = "g1"

    # use the same DoF config as BeyondMimic
    obs_dof: G1BeyondMimicDoF = G1BeyondMimicDoF()
    action_dof: G1BeyondMimicDoF = obs_dof

    # default action scales copied from G1BeyondMimicPolicyCfg
    action_scales: list[float] = [
        *[0.548, 0.548, 0.548, 0.351, 0.351, 0.439, 0.548, 0.548, 0.439, 0.351, 0.351],
        *[0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439],
        *[0.439, 0.439, 0.439, 0.439, 0.075, 0.075, 0.075, 0.075],
    ]

    without_state_estimator: bool = True
    override_robot_anchor_pos: bool = True

    policy_name: str = "latest"  # map to assets/models/g1/promptmimic/latest.onnx

