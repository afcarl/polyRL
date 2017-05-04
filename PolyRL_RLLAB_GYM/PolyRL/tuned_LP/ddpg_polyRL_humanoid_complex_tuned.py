from rllab.algos.ddpg_polyRL import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.persistence_length_higher_dimensions import Persistence_Length_Exploration


from rllab.envs.mujoco.humanoid_env import HumanoidEnv
env = normalize(HumanoidEnv())


print ("State Space", env.observation_space)
print ("Action Space", env.action_space)


"""
PolyRL Hyperparameters
"""
L_p_param = [0.008]
b_step_size = [0.0004]
sigma_param = [0.1]

# L_p_param = [0.004, 0.04, 0.08, 0.16, 0.2, 0.4, 0.8, 2, 4]



num_episodes = 3000
steps_per_episode = 1000
max_exploratory_steps_iters = 20
batch_size_value = 64



for l_p_ind in range(len(L_p_param)):

    for b_ind in range(len(b_step_size)):

        for s_ind in range(len(sigma_param)):

            def run_task(*_):

                env = normalize(HumanoidEnv())


                policy = DeterministicMLPPolicy(
                    env_spec=env.spec,
                    # The neural network policy should have two hidden layers, each with 32 hidden units.
                    hidden_sizes=(32, 32)
                )

                es = OUStrategy(env_spec=env.spec)

                qf = ContinuousMLPQFunction(
                    env_spec=env.spec,
                    hidden_sizes=(32, 32)
                )


                """
                Persistence Length Exploration
                """
                lp = Persistence_Length_Exploration(
                    env=env, 
                    qf=qf, 
                    policy=policy,
                    L_p=L_p_param[l_p_ind],
                    b_step_size=b_step_size[b_ind], 
                    sigma = sigma_param[s_ind],
                    max_exploratory_steps = max_exploratory_steps_iters,
                    batch_size=batch_size_value,
                    n_epochs=num_episodes,
                    scale_reward=0.01,
                    epoch_length=steps_per_episode,
                    qf_learning_rate=0.001,
                    policy_learning_rate=0.0001,
                    )


                """
                DDPG
                """

                algo = DDPG(
                    env=env,
                    policy=policy,
                    es=es,
                    qf=qf,
                    lp=lp,
                    batch_size=batch_size_value,
                    max_path_length=100,
                    epoch_length=steps_per_episode,
                    min_pool_size=10000,
                    n_epochs=num_episodes,
                    discount=0.99,
                    scale_reward=0.01,
                    qf_learning_rate=0.001,
                    policy_learning_rate=0.0001,
                    # Uncomment both lines (this and the plot parameter below) to enable plotting
                    # plot=True,
                )
                algo.train()

            run_experiment_lite(
                run_task,
                # Number of parallel workers for sampling
                n_parallel=1,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                exp_name="PolyRL_DDPG_Humanoid_Complex_Tuned_LP/" + "Parameter_" + str(L_p_param[l_p_ind]),
                seed=1,
                # plot=True,
            )
