from rllab.algos.ddpg_polyRL import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.persistence_length_higher_dimensions import Persistence_Length_Exploration

from rllab.envs.mujoco.hopper_env import HopperEnv
env = normalize(HopperEnv())

print ("State Space", env.observation_space.shape)
print ("Action Space", env.action_space.shape)


def run_task(*_):
    env = normalize(HopperEnv())

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
        L_p=0.08,
        b_step_size=0.0004, 
        sigma = 0.1,
        max_exploratory_steps = 12,
        batch_size=32,
        n_epochs=1000,
        scale_reward=0.01,
        epoch_length=1000,
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
        batch_size=64,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=1000,
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
    exp_name="PolyRL_DDPG_Hopper",
    seed=1,
    # plot=True,
)
