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
from random import randint
from rllab.pool.simple_pool import SimpleReplayPool

from numpy import linalg as LA


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError


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
        max_exploratory_steps = 12, 
        epoch_length=1000,
        length_polymer_chain=20000,
        batch_size=32,
        max_path_length=1000,
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
        scale_reward = 1.0):

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

        chain_actions = np.array([self.initial_action])
        chain_states = np.array([self.initial_state])

        action_trajectory_chain= 0
        state_trajectory_chain = 0

        end_traj_action = 0
        end_traj_state = 0
        itr = 0
        terminal = False
        initial = False



        H_theta = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  )
        H_vector = np.random.normal(H_theta, self.sigma, 6)

        H = ( self.b_step_size / LA.norm(H_vector) ) * H_vector

        all_H = np.array([H])
        all_theta = np.array([])



        sample_policy = pickle.loads(pickle.dumps(self.policy))
        self.initial_action = self.env.action_space.sample()
        last_action_chosen = self.initial_action

        for itr in range(self.max_exploratory_steps):

            print ("LP Exploration Episode", itr)
            print ("Replay Buffer Sample Size", pool.size)

            H_theta = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  )
            H_vector = np.random.normal(H_theta, self.sigma, 6)

            H = ( self.b_step_size / LA.norm(H_vector) ) * H_vector


            next_action = last_action_chosen + H


            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):

                one_vec_x1 = random.randint(-1, 1)
                if one_vec_x1 == 0:
                    one_vec_x1 = 1

                one_vec_x2 = random.randint(-1, 1)
                if one_vec_x2 == 0:
                    one_vec_x2 = 1

                one_vec_x3 = random.randint(-1, 1)
                if one_vec_x3 == 0:
                    one_vec_x3 = 1

                one_vec_x4 = random.randint(-1, 1)
                if one_vec_x4 == 0:
                    one_vec_x4 = 1

                one_vec_x5 = random.randint(-1, 1)
                if one_vec_x5 == 0:
                    one_vec_x5 = 1


                one_vector = np.array([one_vec_x1, one_vec_x2, one_vec_x3, one_vec_x4, one_vec_x5])
                theta_mean = np.arccos( np.exp(   np.true_divide(-self.b_step_size, self.L_p) )  ) * one_vector
                sigma_iden = self.sigma**2*np.identity(5)

                eta = np.random.multivariate_normal(theta_mean, sigma_iden)
                eta = np.concatenate((np.array([0]), eta), axis=0)


                """
                TO DO
                """

                """
                Map H_t to Spherical coordinate
                """
                H_conversion = self.cart2pol(H)

                H = H_conversion + eta


                """
                Map H_t to Cartesian coordinate
                """
                H_conversion = self.pol2cart(H)

                H = H_conversion

                phi_t = next_action


                phi_t_1 = phi_t + H

                chosen_action = np.array([phi_t_1])
                chain_actions = np.append(chain_actions, chosen_action, axis=0)


                """
                Obtained rewards in Swimmer are scaled by 100 
                - also make sure same scaling is done in DDPG without polyRL algo
                """
                chosen_state, reward, terminal, _ = self.env.step(chosen_action)

                ### if an invalid state is reached
                # if terminal == True:
                #     #start a new trajectory during the exploration phase
                #     last_action_chosen = self.env.action_space.sample()

                #     """
                #     Add the tuple of invalid state, a and r(with large penalty negative reward to replay buffer)
                #     put negative reward of -1
                #     """
                #     print ("************Reward at Invalid State", reward)
                #     print ("************Number of Steps before invalid state", epoch_itr)
                #     break

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

                if not terminal and path_length >= self.max_path_length:

                    terminal = True
                    initial = True
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
                        pool.add_sample(state, action, reward * self.scale_reward, terminal, initial)
                    break

                else:
                    initial = False
                    pool.add_sample(state, action, reward * self.scale_reward, terminal, initial)
        

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


        """
        For Walker
        """

        df_a = pd.DataFrame(action_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6'])
        df_a.to_csv("/Users/Riashat/Documents/PhD_Research/RLLAB_Gym/rllab/examples/Action_Chains/exploratory_action_v1_6D_Space_Walker.csv")

        df_s = pd.DataFrame(state_trajectory_chain, columns=['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6', 'Dim 7', 'Dim 8', 'Dim 9', 'Dim 10', 'Dim 11', 'Dim 12', 'Dim 13', 'Dim 14', 'Dim 15', 'Dim 16', 'Dim 17', 'Dim 18', 'Dim 19', 'Dim 20', 'Dim 21'])
        df_s.to_csv("/Users/Riashat/Documents/PhD_Research/RLLAB_Gym/rllab/examples/Action_Chains/exploratory_states_v1_Walker.csv")



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




    def cart2pol(self, cartesian):

        x = cartesian[0]
        y = cartesian[1]
        z = cartesian[2]
        xdot = cartesian[3]
        ydot = cartesian[4]
        zdot = cartesian[5]

        radius = np.sqrt(x**2 + y**2 + z**2)
        alpha = np.arctan2(y, x)



        delta = np.arcsin(z/radius)

        radiusdot = ((x * xdot + y * ydot + z * zdot)) / radius
        alphadot = (x * xdot - y * ydot) / ( (radius * np.cos(delta))**2 )
        deltadot = (zdot - (radiusdot * np.sin(delta))) / (  radius * np.cos(delta) )

        #conditions

        if radius == 0:
            radiusdot = xdot
            alpha, delta, alphadot, deltadot = 0

        if x==0:
            if y < 0:
                alpha = - np.pi / 2
            elif y == 0:
                alpha = 0
            elif y > 0:
                alpha = np.pi / 2

        if np.cos(delta) == 0:
            radiusdot = zdot / np.sin(delta)

            if np.cos(alpha) == 0:
                deltadot = - ydot / (  radius * np.sin(delta) * np.sin(alpha)  )
            else:
                deltadot = - xdot / (  radius * np.sin(delta) * np.cos(alpha)   )


        spherical = np.array([radius, alpha, delta, radiusdot, alphadot, deltadot])


        return spherical



    def pol2cart(self, polar):

        radius = polar[0]
        alpha = polar[1]
        delta = polar[2]
        radiusdot = polar[3]
        alphadot = polar[4]
        deltadot = polar[5]

        x = radius * np.cos(delta) * np.cos(alpha)
        y = radius * np.cos(delta) * np.sin(alpha)
        z = radius * np.sin(delta)

        xdot = - radius * ( alphadot * np.cos(delta) * np.sin(alpha) + deltadot * np.sin(delta) * np.cos(alpha)      )
        xdot = xdot + radiusdot * np.cos(delta) * np.cos(alpha)

        ydot = radius * ( alphadot * np.cos(delta) * np.cos(alpha) - deltadot * np.sin(delta) * np.sin(alpha)   )
        ydot = ydot + radiusdot * np.cos(delta) * np.sin(alpha)

        zdot = radius * deltadot * np.cos(delta) + radiusdot * np.sin(delta)

        cartesian = np.array([x, y, z, xdot, ydot, zdot])

        return cartesian





