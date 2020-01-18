# Reference code: https://github.com/gxnk/reinforcement-learning-code/
import numpy as np
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(6)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        # dimension of action space
        self.n_actions = n_actions
        # dimension of state features
        self.n_features = n_features
        # learning rate
        self.lr = learning_rate
        # reward decay rate
        self.gamma = reward_decay
        # observation values, action values and reward values of a trajectory
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        # build policy net
        self._build_net()
        # start a default session
        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        # initialize variables
        self.sess.run(tf.global_variables_initializer())
    # build policy net implementation
    def _build_net(self):
        with tf.name_scope('input'):
            # create placeholder as input
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # first layer
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1',
        )
        # second layer
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'

        )
        # softmax to get probability of each action
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')
        # define loss function
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob*self.tf_vt)
        # define training, update parameters
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    # define how to choose action, i.e. sample action at state s based on current probability distribution of actions
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs:observation[np.newaxis,:]})
        # sample action given probability distribution
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
    def greedy(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.argmax(prob_weights.ravel())
        return action
    # store transition(state, action, reward)
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    # learn to update parameters of policy network, learn after every episode
    def learn(self):
        # calculate discounted normal reward of each episode
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # call training function to update parameters
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
        })
        # clear data after episode
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm
    def _discount_and_norm_rewards(self):
        # discounted episode rewards
        discounted_ep_rs =np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # normalize
        discounted_ep_rs-= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

