from wheel_legged_gym.envs.balio_vmc.balio_vmc_config import (
    BalioVMCCfg,
    BalioVMCCfgPPO
)

class BalioVMCAdvancedCfg(BalioVMCCfg):
    class env(BalioVMCCfg.env):
        num_privileged_obs = BalioVMCCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3
        num_obs_history = BalioVMCCfg.env.num_observations * BalioVMCCfg.env.obs_history_length
        num_base_vel = 3  ####### resive
        num_decoder = 64  ####### resive
        num_privileged = 64  ####### resive

    class init_state(BalioVMCCfg.init_state):
        pos = [0.0, 0.0, 0.15] # x,y,z [m]
        default_joint_angles = { # target angles when action = 0.0
            "lf0_Joint": 0.56,
            "lf1_Joint": 1.12,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": 0.56,
            "rf1_Joint": 1.12,
            "r_wheel_Joint": 0.0,
        }

    class control(BalioVMCCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0
        pos_action_scale = 0.5
        vel_action_scale = 10.0

############################################################
        l0_offset = 0.145
        feedforward_force = 17.5  # [N]#TODO: 40.0

        kp_theta = 10.0 #TODO:50.0  # [N*m/rad]
        kd_theta = 1.0 #TODO:3.0  # [N*m*s/rad]
        kp_l0 = 300.0 #TODO:900.0  # [N/m]
        kd_l0 = 8.0 #TODO:20.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.5}  # [N*m*s/rad]
############################################################

    class normalization(BalioVMCCfg.normalization):
        class obs_scales(BalioVMCCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25
            # wheel pos should be zero!
            dof_pos = 0.0

    class noise(BalioVMCCfg.noise):
        class noise_scales(BalioVMCCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1

    class asset(BalioVMCCfg.asset):
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/balio/urdf/balio_issac_gym.urdf"
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/balio/urdf/balio_issac_gym_new.urdf"
        name = "Balio"
        offset = 0.035
        l1 = 0.1 #TODO:0.15
        l2 = 0.165 #TODO:0.25
        penalize_contacts_on = ["lf", "rf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False

class BalioVMCAdvancedCfgPPO(BalioVMCCfgPPO):
    class policy(BalioVMCCfgPPO.policy):
        lstm_hidden_dims = [128,64]
        vae_hidden_dims = [256, 128, 64]  ####### resive
        privileged_hidden_dims = [256, 128, 64]  ####### resive

    class algorithm(BalioVMCCfgPPO.algorithm):
        ppo_name = "PPO_ASYMMETRIC" ####### resive
        # ppo_name = "PPO_SEQUENCE"   ####### resive
        kl_decay = (BalioVMCCfgPPO.algorithm.desired_kl - 0.002) / BalioVMCCfgPPO.runner.max_iterations

    class runner(BalioVMCCfgPPO.runner):
        # logging
        experiment_name = "balio_vmc_advanced"
        max_iterations = 3000
        # TODO:
        policy_class_name = "ActorCriticAsymmetric"

