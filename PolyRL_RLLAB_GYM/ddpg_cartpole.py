from rllab.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction




def run_task(*_):


    """
    DPG on Swimmer environment
    """
    env = normalize(CartpoleEnv())

    """
    Initialise the policy as a neural network policy
    """
    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )


    """
    Defining exploration strategy : OUStrategy - 
    """
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """
    es = OUStrategy(env_spec=env.spec)


    """
    Defining the Q network
    """
    qf = ContinuousMLPQFunction(env_spec=env.spec)

    """
    Using the DDPG algorithm
    """
    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=32,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=100,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        #Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )


    """
    Training the networks based on the DDPG algorithm
    """
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True,
)

