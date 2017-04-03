""" 
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn

from replay_buffer import ReplayBuffer

from numpy import linalg as LA

from lib import plotting
import pandas as pd

from numpy.linalg import inv

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


# ==========================
#   Training Parameters
# ==========================
# Max training steps
# MAX_EPISODES = 50000

MAX_EPISODES = 2000

# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'HalfCheetah-v1'

MONITOR_DIR = './results/gym_ddpg'


RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000


#ORIGINAL
MINIBATCH_SIZE = 64


# MINIBATCH_SIZE = 128

# ===========================
#   Actor and Critic DNNs
# ===========================


# action_examples = np.array([ENV_NAME.action_space.sample() for x in range(10000)])

action_examples = np.random.uniform(-2.0, 2.0,1)

scaler_action = sklearn.preprocessing.StandardScaler()
scaler_action.fit(action_examples)


# featurizer_action = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#         ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#         ])
# featurizer_action.fit(scaler_action.transform(action_examples))



featurizer_action = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=1)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=1))
        ])
featurizer_action.fit(scaler_action.transform(action_examples))



def featurize_action(action):
    # action = np.array([action])
    scaled = scaler_action.transform([action])
    featurized_action = featurizer_action.transform(scaled)
    return featurized_action[0]





class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2s
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient})


    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={self.inputs: inputs})



    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars





class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={self.target_inputs: inputs,  self.target_action: action})

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars




# ===========================
#  LP Exploration
# ===========================


def LP_Exploration(env, action, state, actor, critic, length_polymer_chain, L_p, b_step_size, sigma, replay_buffer, ep_ave_max_q):


    chain_actions = np.array([action])

    chain_states = state


    #draw theta from a Gaussian distribution
    theta_mean = np.arccos( np.exp(   np.true_divide(-b_step_size, L_p) )  )
    theta = np.random.normal(theta_mean, sigma, 1)

    action_trajectory_chain= 0
    state_trajectory_chain = 0

    end_traj_action = 0
    end_traj_state = 0

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    operator = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),np.cos(theta)], [np.cos(theta), -np.sin(theta)]  ]).reshape(6,6)


    print "Operator", operator

    print "Operator Shape", operator.shape

    print X


    #building the polymer chain
    while True:

        coin_flip = np.random.randint(2, size=1)

        if coin_flip == 0:
            operator = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta),  np.cos(theta)]]).reshape(6,6)
        elif coin_flip == 1:
            operator = np.array([[np.cos(theta), np.sin(theta)], [- np.sin(theta),  np.cos(theta)]]).reshape(6,6)



        phi_t = action

        phi_t_1 = phi_t + np.dot(operator, phi_t)

        chosen_action = np.array([phi_t_1])
        
        chain_actions = np.append(chain_actions, chosen_action, axis=0)


        chosen_state, reward, terminal, _ = env.step(chosen_action)
 
        chain_states = np.append(chain_states, chosen_state)    



        replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(chosen_action, (actor.a_dim,)), reward, terminal, np.reshape(chosen_state, (actor.s_dim,)))

        if terminal:
            chosen_state = env.reset()



        if replay_buffer.size() > MINIBATCH_SIZE:
            s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

            target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))



            y_i = []
            for k in xrange(MINIBATCH_SIZE):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + GAMMA * target_q[k])


            #### PROBEM : Probel with A as feature vector is here
            predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

            ep_ave_max_q += np.amax(predicted_q_value)

            a_outs = actor.predict(s_batch)
            grads = critic.action_gradients(s_batch, a_outs)
            actor.train(s_batch, grads[0])

            actor.update_target_network()
            critic.update_target_network()


        if replay_buffer.size() > length_polymer_chain:
            end_traj_action = chosen_action
            end_traj_state = chosen_state
            break



    action_trajectory_chain = chain_actions
    state_trajectory_chain = chain_states


    return action_trajectory_chain, state_trajectory_chain, end_traj_action, end_traj_state


# ===========================
#   Agent Training
# ===========================


def train(sess, env, actor, critic, length_polymer_chain, L_p, b_step_size, sigma):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)


    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(MAX_EPISODES),
        episode_rewards=np.zeros(MAX_EPISODES))  


    for i in xrange(MAX_EPISODES):

        print "Number of Episode", i

        s = env.reset()
        initial_action = env.action_space.sample()

        ep_reward = 0
        ep_ave_max_q = 0

        """
        LP Exploration 
        """
        action_trajectory_chain, state_trajectory_chain, end_traj_action, end_traj_state = LP_Exploration(env, initial_action, s, actor, critic, length_polymer_chain, L_p, b_step_size, sigma, replay_buffer, ep_ave_max_q)

        s = end_traj_state
        a = end_traj_action


        for j in xrange(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            # Added exploration noise
            ### this is the usual exploration as done in DDPG
            ### we still do this? Or choose next a based on L_p exploration?
            """
            CHECK THIS
            """
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + (1. / (1. + i))



            s2, r, terminal, info = env.step(a)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:

                #### Sampling a random minibatch from the buffer
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                # calculate the target
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])


                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])


                # Update target networks
                actor.update_target_network()
                critic.update_target_network()


            s = s2
            ep_reward += r

            stats.episode_rewards[i] += r

            if terminal:

                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: ep_reward,
                #     summary_vars[1]: ep_ave_max_q / float(j)
                # })

                # writer.add_summary(summary_str, i)
                # writer.flush()

                print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j))

                break


    return stats





def main(_):
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        print "Action Space", env.action_space.shape

        print "State Space", env.observation_space.shape

        # Ensure action bound is symmetric
        #assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())


        length_polymer_chain = 200
        L_p = 200
        b_step_size = 1
        sigma = 0.5

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

    
        stats = train(sess, env, actor, critic, length_polymer_chain, L_p, b_step_size, sigma)

        rewards_polyddpg = pd.Series(stats.episode_rewards).rolling(1, min_periods=1).mean()    
        cum_rwd = rewards_polyddpg

        np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Persistence_Length_Exploration/Results/'  + 'PolyRL_DDPG_v2_HalfCheetah' + '.npy', cum_rwd)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()