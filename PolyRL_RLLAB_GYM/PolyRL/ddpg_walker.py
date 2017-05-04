from rllab.algos.ddpg import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


from rllab.envs.mujoco.walker2d_env import Walker2DEnv

env = normalize(Walker2DEnv())


# H_layer_first = [32, 100, 400]
# H_layer_second = [32, 100, 300]

H_layer_first = [32]
H_layer_second = [32]

reward_scaling = [0.01, 0.1, 1.0]

critic_learning_rate = [1e-3, 10e-3]
actor_learning_rate = [1e-4, 10e-4]

#0.99 was originally set by rllab
discount_factor = 0.99

#originally : 32 set by rllab
size_of_batch = 64

number_of_episodes = 3000

for h in range(len(H_layer_first)):
    print("Network Structure, First Layer", H_layer_first[h])
    print("Network Structure, Second Layer", H_layer_second[h])


    for r in range(len(reward_scaling)): 
        print ("Reward Scaling Factor", reward_scaling[r])


        for c in range(len(critic_learning_rate)):
            print ("Critic Learning Rate", critic_learning_rate[c])

            def run_task(*_):
                env = normalize(Walker2DEnv())

                policy = DeterministicMLPPolicy(
                    env_spec=env.spec,
                    # The neural network policy should have two hidden layers, each with 32 hidden units.
                    hidden_sizes=(H_layer_first[h], H_layer_second[h])
                )

                es = OUStrategy(env_spec=env.spec)

                qf = ContinuousMLPQFunction(env_spec=env.spec)

                algo = DDPG(
                    env=env,
                    policy=policy,
                    es=es,
                    qf=qf,
                    batch_size=size_of_batch,
                    max_path_length=100,
                    epoch_length=1000,
                    min_pool_size=10000,
                    n_epochs=number_of_episodes,
                    discount=discount_factor,
                    scale_reward=reward_scaling[r],
                    qf_learning_rate=critic_learning_rate[c],
                    policy_learning_rate=actor_learning_rate[c],
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
                exp_name="DDPG_Walker/" + "Walker",
                seed=1,
                # plot=True,
            )
