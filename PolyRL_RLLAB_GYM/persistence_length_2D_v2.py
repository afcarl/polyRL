from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import pyprind

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special
from rllab.misc import ext
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from functools import partial
import rllab.misc.logger as logger
import theano.tensor as TT
import pickle as pickle
import numpy as np
import pyprind
import lasagne
import rllab.misc.logger as logger

import random 
import pandas as pd 


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError



class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size


class Persistence_Length_Exploration(ExplorationStrategy, Serializable):
    """
    Persistence Length Based Exploration.
    """


    """
    Need to change params - max_exploratory_steps, min_pool_size, epoch_length etc etc
    """
    def __init__(
        self, 
        env, 
        qf, 
        policy,
        L_p=0.08, 
        b_step_size=0.0004, 
        sigma = 0.1, 
        max_exploratory_steps = 20, 
        epoch_length=2000,
        length_polymer_chain=100000,
        batch_size=32,
        max_path_length=200000,
        qf_weight_decay=0.,
        qf_update_method='adam',
        qf_learning_rate=1e-3,
        policy_weight_decay=0,
        policy_update_method='adam',
        policy_learning_rate=1e-3, 
        soft_target=True,
        soft_target_tau=0.001,
        min_pool_size=10000,
        replay_pool_size=1000000,
        discount=0.99,
        n_updates_per_sample=1, 
        include_horizon_terminal_transitions=False, 
        scale_reward = 100.0):

        self.env = env
        assert isinstance(self.env.action_space, Box)
        assert len(self.env.action_space.shape) == 1       
        Serializable.quick_init(self, locals())

        self.policy = policy
        self.qf = qf

        self.replay_pool_size = replay_pool_size
        self.batch_size = batch_size
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            parse_update_method(
                qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = \
            parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate,
            )
        self.policy_learning_rate = policy_learning_rate
        self.discount = discount
        self.soft_target_tau = soft_target_tau


        self.b_step_size = b_step_size
        self.L_p = L_p
        self.sigma = sigma
        self.max_exploratory_steps = max_exploratory_steps
        self.epoch_length = epoch_length

        self.length_polymer_chain = self.max_exploratory_steps*self.epoch_length

        self.initial_action = self.env.action_space.sample()
        self.initial_state = self.env.reset()

        self.min_pool_size = min_pool_size
        self.n_updates_per_sample = n_updates_per_sample
        self.max_path_length = max_path_length
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.scale_reward = scale_reward
        self._action_space = self.env.action_space

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []



    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        # if self.plot:
        #     plotter.init_plot(self.env, self.policy)


    @overrides
    def lp_exploration(self):

        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim,
        )

        self.start_worker()
        self.init_opt()

        path_length=0

        self.initial_action = self.env.action_space.sample()
        self.b_step_size = np.linalg.norm(self.initial_action)

        chain_actions = np.array([self.initial_action])
        chain_states = np.array([self.initial_state])

        action_trajectory_chain= 0
        state_trajectory_chain = 0

        end_traj_action = 0
        end_traj_state = 0
        itr = 0
        terminal = False

        sample_policy = pickle.loads(pickle.dumps(self.policy))
        last_action_chosen = self.initial_action

        all_theta = np.array([])

        for itr in range(self.max_exploratory_steps):

            print ("LP Exploration Episode", itr)
            print ("Replay Buffer Sample Size", pool.size)

            # next_action = np.dot(operator, last_action_chosen)
            next_action = last_action_chosen

            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):

                theta_mean = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  )
                theta = np.random.normal(theta_mean, self.sigma, 1)   
                all_theta = np.append(all_theta, theta)

                coin_flip = np.random.randint(2, size=1)
                if coin_flip == 0:
                    operator = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta),  np.cos(theta)]]).reshape(2,2)
                elif coin_flip == 1:
                    operator = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta),  np.cos(theta)]]).reshape(2,2)


                phi_t = next_action
                phi_t_1 = np.dot(operator, phi_t)

                chosen_action = np.array([phi_t_1])
                chain_actions = np.append(chain_actions, chosen_action, axis=0)

                """
                Obtained rewards in Swimmer are scaled by 100 
                - also make sure same scaling is done in DDPG without polyRL algo
                """
                chosen_state, reward, terminal, _ = self.env.step(chosen_action)

                ### if an invalid state is reached
                if terminal == True:
                    #start a new trajectory during the exploration phase
                    last_action_chosen = self.env.action_space.sample()

                    """
                    Add the tuple of invalid state, a and r(with large penalty negative reward to replay buffer)
                    put negative reward of -1
                    """
                    print ("************Reward at Invalid State", reward)
                    print ("************Number of Steps before invalid state", epoch_itr)
                    break

                chain_states = np.append(chain_states, np.array([chosen_state]), axis=0)    


                action = chosen_action
                state = chosen_state

                end_traj_state = chosen_state
                end_traj_action = chosen_action

                next_action = phi_t_1

                path_length += 1

                if not terminal and path_length >= self.max_path_length:

                    terminal = True
                    terminal_state = chosen_state

                    print ("LP Epoch Length Terminated")

                    path_length = 0
                    path_return = 0
                    state = self.env.reset()
                    sample_policy.reset()

                    print ("Step Number at Terminal", epoch_itr)

                    # only include the terminal transition in this case if the flag was set
                    if self.include_horizon_terminal_transitions:
                        #### adding large negative reward to the terminal state reward??? Check this
                        pool.add_sample(state, action, reward * self.scale_reward*100, terminal)
                    break

                else:
                    pool.add_sample(state, action, reward * self.scale_reward, terminal)
        

                if pool.size >= self.min_pool_size:
                    for update_itr in range(self.n_updates_per_sample):
                        # Train policy
                        batch = pool.random_batch(self.batch_size)
                        updated_q_network = self.do_training(itr, batch)
                        sample_policy.set_param_values(self.policy.get_param_values())

                itr += 1

            last_action_chosen = action
            last_action_chosen = last_action_chosen[0, :]



        action_trajectory_chain = chain_actions
        state_trajectory_chain = chain_states
        end_trajectory_action = end_traj_action
        end_trajectory_state = end_traj_state



        df_a = pd.DataFrame(action_trajectory_chain, columns=['Dim 1', 'Dim 2'])
        df_a.to_csv("/Users/Riashat/Documents/PhD_Research/RLLAB_Gym/rllab/examples/Action_Chains/exploratory_action_v2.csv")

        df_s = pd.DataFrame(state_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6', 'Dim 7', 'Dim 8', 'Dim 9', 'Dim 10', 'Dim 11', 'Dim 12', 'Dim 13'])
        df_s.to_csv("/Users/Riashat/Documents/PhD_Research/RLLAB_Gym/rllab/examples/Action_Chains/exploratory_states_v2.csv")

        # df = pd.DataFrame({'Theta Values': all_theta,
        #            'b_step_size': self.b_step_size)

        df = pd.DataFrame(all_theta, columns=['Theta values'])
        df.to_csv("/Users/Riashat/Documents/PhD_Research/RLLAB_Gym/rllab/examples/Action_Chains/All Exploration Params_v2.csv")


        return updated_q_network, action_trajectory_chain, state_trajectory_chain, end_trajectory_action, end_trajectory_state


    def init_opt(self):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(self.policy))
        target_qf = pickle.loads(pickle.dumps(self.qf))

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([TT.sum(TT.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([TT.sum(TT.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -TT.mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_updates = self.qf_update_method(
            qf_reg_loss, self.qf.get_params(trainable=True))
        policy_updates = self.policy_update_method(
            policy_reg_surr, self.policy.get_params(trainable=True))

        f_train_qf = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=[qf_loss, qval],
            updates=qf_updates
        )

        f_train_policy = ext.compile_function(
            inputs=[obs],
            outputs=policy_surr,
            updates=policy_updates
        )

        self.opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )



    def do_training(self, itr, batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.opt_info["target_qf"]
        target_policy = self.opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals

        f_train_qf = self.opt_info["f_train_qf"]
        f_train_policy = self.opt_info["f_train_policy"]

        qf_loss, qval = f_train_qf(ys, obs, actions)

        policy_surr = f_train_policy(obs)


        target_policy.set_param_values(
            target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
            self.policy.get_param_values() * self.soft_target_tau)


        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)


        """
        Check this again - whether Q network is actually updated or not
        """



        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

        q_network_exploratory_update = self.qf

        return q_network_exploratory_update






