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

from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import random 
import pandas as pd 
from random import randint
from numpy import linalg as LA



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
    PolyRL Exploration Strategy
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            L_p=0.08, 
            b_step_size=0.0004, 
            sigma = 0.1, 
            max_exploratory_steps = 20, 
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            max_path_length=250,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            eval_samples=10000,
            soft_target=True,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False,
            pause_for_plot=False):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :param pause_for_plot: Whether to pause before continuing when plotting.
        :return:
        """
        self.env = env
        self.policy = policy
        self.qf = qf
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.max_path_length = max_path_length
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
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot
        self.pause_for_plot = pause_for_plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

        self.opt_info = None


        """
        PolyRL Hyperparameters
        """
        self.b_step_size = b_step_size
        self.L_p = L_p
        self.sigma = sigma
        self.max_exploratory_steps = max_exploratory_steps




    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def lp_exploration(self):
        # This seems like a rather sequential method
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim,
        )
        self.start_worker()


        self.init_opt()
        itr = 0
        path_length = 0
        path_return = 0
        terminal = False

        observation = self.env.reset()
        self.initial_action = self.env.action_space.sample()

        sample_policy = pickle.loads(pickle.dumps(self.policy))

        chain_actions = np.array([self.initial_action])
        chain_states = np.array([observation])

        action_trajectory_chain= 0
        state_trajectory_chain = 0

        end_traj_action = 0
        end_traj_state = 0

        H_theta = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  )
        H_vector = np.random.normal(H_theta, self.sigma, self.env.action_space.shape[0])

        H = ( self.b_step_size / LA.norm(H_vector) ) * H_vector

        all_H = np.array([H])
        all_theta = np.array([])


        last_action_chosen = self.initial_action


        for epoch in range(self.max_exploratory_steps):

            print ("LP Exploration Episode", epoch)
            print ("Replay Buffer Sample Size", pool.size)

            if epoch==0:
                next_action = last_action_chosen + H
            else:
                next_action = last_action_chosen


            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):

                if self.env.action_space.shape[0] == 6:

                    one_vec_x1 = random.randint(-1, 0)
                    if one_vec_x1 == 0:
                        one_vec_x1 = 1

                    one_vec_x2 = random.randint(-1, 0)
                    if one_vec_x2 == 0:
                        one_vec_x2 = 1

                    one_vec_x3 = random.randint(-1, 0)
                    if one_vec_x3 == 0:
                        one_vec_x3 = 1

                    one_vec_x4 = random.randint(-1, 0)
                    if one_vec_x4 == 0:
                        one_vec_x4 = 1

                    one_vec_x5 = random.randint(-1, 0)
                    if one_vec_x5 == 0:
                        one_vec_x5 = 1

                    one_vector = np.array([one_vec_x1, one_vec_x2, one_vec_x3, one_vec_x4, one_vec_x5])

                elif self.env.action_space.shape[0] == 3:

                    one_vec_x1 = random.randint(-1, 0)
                    if one_vec_x1 == 0:
                        one_vec_x1 = 1

                    one_vec_x2 = random.randint(-1, 0)
                    if one_vec_x2 == 0:
                        one_vec_x2 = 1
                    one_vector = np.array([one_vec_x1, one_vec_x2])


                theta_mean = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  ) * one_vector
                sigma_iden = self.sigma**2*np.identity(self.env.action_space.shape[0]-1)

                eta = np.random.multivariate_normal(theta_mean, sigma_iden)
                eta = np.concatenate((np.array([0]), eta), axis=0)

                """
                Map H_t to Spherical coordinate
                """
                if self.env.action_space.shape[0] == 6:
                    H_conversion = self.cart2pol_6D(H)
                elif self.env.action_space.shape[0] ==3:
                    H_conversion = self.cart2pol_3D(H)
                H = H_conversion + eta


                """
                Map H_t to Cartesian coordinate
                """
                if self.env.action_space.shape[0] == 6:
                    H_conversion = self.pol2cart_6D(H)
                elif self.env.action_space.shape[0] ==3:
                    H_conversion = self.pol2cart_3D(H)                    
                H = H_conversion


                phi_t = next_action
                phi_t_1 = phi_t + H

                chosen_action = np.array([phi_t_1])
                chain_actions = np.append(chain_actions, chosen_action, axis=0)

                chosen_state, reward, terminal, _ = self.env.step(chosen_action)
                chain_states = np.append(chain_states, np.array([chosen_state]), axis=0)    

                action = chosen_action
                state = chosen_state
                end_traj_state = chosen_state
                end_traj_action = chosen_action

                #updates to be used in next iteration
                H = phi_t_1 - phi_t
                all_H = np.append(all_H, np.array([H]), axis=0)
                next_action = phi_t_1

                path_length += 1

                # Execute policy
                # if terminal:  # or path_length > self.max_path_length:
                #     # Note that if the last time step ends an episode, the very
                #     # last state and observation will be ignored and not added
                #     # to the replay pool
                #     observation = self.env.reset()
                #     self.es.reset()
                #     sample_policy.reset()
                #     self.es_path_returns.append(path_return)
                #     path_length = 0
                #     path_return = 0
                    

                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    terminal_state = chosen_state

                    path_length = 0
                    path_return = 0
                    state = self.env.reset()
                    sample_policy.reset()
                    
                    print ("LP Epoch Length Terminated")
                    print ("Step Number at Terminal", epoch_itr)

                    # only include the terminal transition in this case if the flag was set
                    if self.include_horizon_terminal_transitions:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal)

                else:
                    pool.add_sample(observation, action, reward * self.scale_reward, terminal)

                observation = state
                #observation = next_observation

                if pool.size >= self.min_pool_size:
                    for update_itr in range(self.n_updates_per_sample):
                        # Train policy
                        batch = pool.random_batch(self.batch_size)
                        updated_q_network, updated_policy_network = self.do_training(itr, batch)
                    sample_policy.set_param_values(self.policy.get_param_values())

                itr += 1

            last_action_chosen = action
            last_action_chosen = last_action_chosen[0, :]


        action_trajectory_chain = chain_actions
        state_trajectory_chain = chain_states
        end_trajectory_action = end_traj_action
        end_trajectory_state = end_traj_state




        if self.env.action_space.shape[0]==6:
            df_a = pd.DataFrame(action_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6'])
            df_a.to_csv("/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/data/local/experiment/PolyRL_DDPG_Walker/walker_action_chain.csv")

            df_s = pd.DataFrame(state_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6', 'Dim 7', 'Dim 8', 'Dim 9', 'Dim 10', 'Dim 11', 'Dim 12', 'Dim 13', 'Dim 14', 'Dim 15', 'Dim 16', 'Dim 17', 'Dim 18', 'Dim 19', 'Dim 20', 'Dim 21'])
            df_s.to_csv("/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/data/local/experiment/PolyRL_DDPG_Walker/walker_state_chain.csv")

        elif self.env.action_space.shape[0]==3:
            df_a = pd.DataFrame(action_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3'])
            df_a.to_csv("/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/data/local/experiment/PolyRL_DDPG_Hopper/hopper_action_chain.csv")

            df_s = pd.DataFrame(state_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6', 'Dim 7', 'Dim 8', 'Dim 9', 'Dim 10', 'Dim 11', 'Dim 12', 'Dim 13', 'Dim 14', 'Dim 15', 'Dim 16', 'Dim 17', 'Dim 18', 'Dim 19', 'Dim 20'])
            df_s.to_csv("/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/data/local/experiment/PolyRL_DDPG_Hopper/hopper_state_chain.csv")

        return updated_q_network, updated_policy_network, action_trajectory_chain, state_trajectory_chain, end_trajectory_action, end_trajectory_state




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

        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)
        self.q_averages.append(qval)
        self.y_averages.append(ys)
        
        q_network_exploratory_update = self.qf
        policy_network_exploratory_update = self.policy

        return q_network_exploratory_update, policy_network_exploratory_update




    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        policy_reg_param_norm = np.linalg.norm(
            self.policy.get_param_values(regularizable=True)
        )
        qfun_reg_param_norm = np.linalg.norm(
            self.qf.get_param_values(regularizable=True)
        )

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('AverageReturn',
                              np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('AverageAction', average_action)

        logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)
        logger.record_tabular('QFunRegParamNorm',
                              qfun_reg_param_norm)

        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.opt_info["target_qf"],
            target_policy=self.opt_info["target_policy"],
            es=self.es,
        )



    def cart2pol_6D(self, cartesian):

        x_1 = cartesian[0]
        x_2 = cartesian[1]
        x_3 = cartesian[2]
        x_4 = cartesian[3]
        x_5 = cartesian[4]
        x_6 = cartesian[5]

        modulus = x_1**2 + x_2**2 + x_3**2 + x_4**2 + x_5**2 + x_6**2

        radius = np.sqrt(modulus)
        phi_1 = np.arccos(x_1 / radius)
        phi_2 = np.arccos(x_2 / radius)
        phi_3 = np.arccos(x_3 / radius)

        phi_4 = np.arccos(x_4 / (np.sqrt( x_4**2 + x_5**2 + x_6**2  )) )

        if x_6 >= 0:
            phi_5 = np.arccos(x_5 / (np.sqrt( x_5**2 + x_6**2 )))
        else:
            phi_5 = (2 * np.pi) - np.arccos(x_5 / (np.sqrt( x_5**2 + x_6**2 )))


        spherical = np.array([radius, phi_1, phi_2, phi_3, phi_4, phi_5])


        return spherical


    def cart2pol_3D(self, cartesian):

        x_1 = cartesian[0]
        x_2 = cartesian[1]
        x_3 = cartesian[2]


        modulus = x_1**2 + x_2**2 + x_3**2 

        radius = np.sqrt(modulus)
        phi_1 = np.arccos(x_1 / radius)
        phi_2 = np.arccos(x_2 / radius)

        spherical = np.array([radius, phi_1, phi_2])


        return spherical



    def pol2cart_6D(self, polar):


        radius = polar[0]
        phi_1 = polar[1]
        phi_2 = polar[2]
        phi_3 = polar[3]
        phi_4 = polar[4]
        phi_5 = polar[5]

        x_1 = radius * np.cos(phi_1)
        x_2 = radius * np.sin(phi_1) * np.cos(phi_2)
        x_3 = radius * np.sin(phi_1) * np.sin(phi_2) * np.cos(phi_3)
        x_4 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.cos(phi_4)
        x_5 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.cos(phi_5)
        x_6 = radius * np.sin(phi_1) * np.sin(phi_2) * np.sin(phi_3) * np.sin(phi_4) * np.sin(phi_5)

        cartesian = np.array([x_1, x_2, x_3, x_4, x_5, x_6])

        return cartesian


    def pol2cart_3D(self, polar):


        radius = polar[0]
        phi_1 = polar[1]
        phi_2 = polar[2]

        x_1 = radius * np.cos(phi_1)
        x_2 = radius * np.sin(phi_1) * np.cos(phi_2)
        x_3 = radius * np.sin(phi_1) * np.sin(phi_2) 

        cartesian = np.array([x_1, x_2, x_3])


        return cartesian


