import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from sklearn.linear_model import LogisticRegression



class STNE(object):

    def __init__(self, sess, args, node_fea=None, node_fea_trainable=False):

        #self.num_items = num_items

        self.node_fea = node_fea
        
        self.fea_dim = self.node_fea.shape[1]
        self.node_num = self.node_fea.shape[0]
        self.seq_len = args.seq_len
        self.hidden_dim = args.hidden_dim
        self.depth = args.depth
        
        self.node_fea_trainable = args.trainable
        self.sess = sess
        self.lr = args.lr
        self.tau = args.tau
        self.dropout = args.dropout

        self.num_other_variables = len(tf.trainable_variables())

        with tf.variable_scope('Active'):
            self.input_seqs, self.output_preds, self.encoder_output = self.create_stne_network_depart("Active")
        self.network_params = tf.trainable_variables()[self.num_other_variables:]
        #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        with tf.variable_scope('Target'):
            self.target_input_seqs, self.target_output_preds, self.target_encoder_output = self.create_stne_network_depart("Target")
        self.target_network_params = tf.trainable_variables()[len(self.network_params) + self.num_other_variables:]
        #print(len(self.target_network_params))
        #print(len(self.network_params))

        # delayed updating recommender network ops
        self.update_target_network_params = \
            [self.target_network_params[i].assign( \
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1 - self.tau)) \
                for i in range(len(self.target_network_params))]

        # network parameters --> target network parameters
        self.assign_target_network_params = \
            [self.target_network_params[i].assign( \
                self.network_params[i]) for i in range(len(self.target_network_params))]

        # target network parameters -->  network parameters
        self.assign_active_network_params = \
            [self.network_params[i].assign( \
                self.target_network_params[i]) for i in range(len(self.network_params))]

        
        reward = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_input_seqs, logits=self.target_output_preds)
        #print(reward.get_shape().as_list())
        self.reward = tf.reduce_sum(reward, axis=1)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_input_seqs, logits=self.target_output_preds)
        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')

        self.global_step = tf.Variable(1, name="global_step", trainable=False)

        self.gradients = tf.gradients(self.loss_ce, self.target_network_params)


        #self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss_ce, global_step=self.global_step)
        self.optimizer = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=1e-8).apply_gradients(
            zip(self.gradients, self.network_params), global_step=self.global_step)
        #self.optimizer = tf.train.RMSPropOptimizer(self.lr).apply_gradients(zip(self.gradients, self.network_params), global_step=self.global_step)
        # total variables
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        self.num_network_params = len(self.network_params)
        self.num_target_network_params = len(self.target_network_params)


    def create_stne_variable(self, scope):

        with tf.name_scope(scope):

            self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
            self.embedding_W = tf.Variable(initial_value=self.node_fea, name='encoder_embed', trainable=self.node_fea_trainable)
            self.input_seq_embed = tf.nn.embedding_lookup(self.embedding_W, self.input_seqs, name='input_embed_lookup')






    def create_stne_inference(self, scope):

        with tf.name_scope(scope):

            # encoder
            encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)
            encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)
            if self.depth == 1:
                encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0])
                encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0])
            else:
                encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)
                encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)

                encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_fw_0] + [encoder_cell_fw_1] * (self.depth - 1))
                encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([encoder_cell_bw_0] + [encoder_cell_bw_1] * (self.depth - 1))

            encoder_outputs, encoder_final = \
            bi_rnn(encoder_cell_fw_all, encoder_cell_bw_all, inputs=self.input_seq_embed, dtype=tf.float32)

            c_fw_list, h_fw_list, c_bw_list, h_bw_list = [], [], [], []

            for d in range(self.depth):
                (c_fw, h_fw) = encoder_final[0][d]
                (c_bw, h_bw) = encoder_final[1][d]
                c_fw_list.append(c_fw)
                h_fw_list.append(h_fw)
                c_bw_list.append(c_bw)
                h_bw_list.append(h_bw)

            decoder_init_state = tf.concat(c_fw_list + c_bw_list, axis=-1), tf.concat(h_fw_list + h_bw_list, axis=-1)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim * 2), output_keep_prob=1 - self.dropout)
            decoder_init_state = LSTMStateTuple(
                tf.layers.dense(decoder_init_state[0], units=self.hidden_dim * 2, activation=None),
                tf.layers.dense(decoder_init_state[1], units=self.hidden_dim * 2, activation=None))
            
            self.encoder_output = tf.concat(encoder_outputs, axis=-1)
            #print(self.encoder_output.get_shape().as_list())
            encoder_output_T = tf.transpose(self.encoder_output, [1, 0, 2])  # h

            new_state = decoder_init_state
            outputs_list = []
            for i in range(self.seq_len):
                new_output, new_state = decoder_cell(tf.zeros(shape=tf.shape(encoder_output_T)[1:]), new_state)  # None
                outputs_list.append(new_output)

            decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
            decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim

            self.output_preds = tf.layers.dense(decoder_outputs, units=self.node_num, activation=None)

    def create_stne_network_depart(self, scope):
        self.create_stne_variable(scope)
        self.create_stne_inference(scope)
        return self.input_seqs, self.output_preds, self.encoder_output


    def create_stne_network(self, scope):

        with tf.name_scope(scope):
            self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
            self.embedding_W = tf.Variable(initial_value=self.node_fea, name='encoder_embed', trainable=self.node_fea_trainable)

            self.input_seq_embed = tf.nn.embedding_lookup(self.embedding_W, self.input_seqs, name='input_embed_lookup')
            # input_seq_embed = tf.layers.dense(input_seq_embed, units=1200, activation=None)

            # encoder
            self.encoder_cell_fw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)
            self.encoder_cell_bw_0 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)
            if self.depth == 1:
                self.encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([self.encoder_cell_fw_0])
                self.encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([self.encoder_cell_bw_0])
            else:
                self.encoder_cell_fw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)
                self.encoder_cell_bw_1 = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim), output_keep_prob=1 - self.dropout)

                self.encoder_cell_fw_all = tf.contrib.rnn.MultiRNNCell([self.encoder_cell_fw_0] + [self.encoder_cell_fw_1] * (self.depth - 1))
                self.encoder_cell_bw_all = tf.contrib.rnn.MultiRNNCell([self.encoder_cell_bw_0] + [self.encoder_cell_bw_1] * (self.depth - 1))

            self.encoder_outputs, self.encoder_final = bi_rnn(self.encoder_cell_fw_all, self.encoder_cell_bw_all, inputs=self.input_seq_embed,
                                                    dtype=tf.float32)

            self.c_fw_list, self.h_fw_list, self.c_bw_list, self.h_bw_list = [], [], [], []
            
            for d in range(self.depth):
                (c_fw, h_fw) = self.encoder_final[0][d]
                (c_bw, h_bw) = self.encoder_final[1][d]
                self.c_fw_list.append(c_fw)
                self.h_fw_list.append(h_fw)
                self.c_bw_list.append(c_bw)
                self.h_bw_list.append(h_bw)

            self.decoder_init_state = tf.concat(self.c_fw_list + self.c_bw_list, axis=-1), tf.concat(self.h_fw_list + self.h_bw_list, axis=-1)
            self.decoder_cell = tf.contrib.rnn.DropoutWrapper(LSTMCell(self.hidden_dim * 2), output_keep_prob=1 - self.dropout)
            self.decoder_init_state = LSTMStateTuple(
                tf.layers.dense(self.decoder_init_state[0], units=self.hidden_dim * 2, activation=None),
                tf.layers.dense(self.decoder_init_state[1], units=self.hidden_dim * 2, activation=None))

            self.encoder_output = tf.concat(self.encoder_outputs, axis=-1)
            self.encoder_output_T = tf.transpose(self.encoder_output, [1, 0, 2])  # h

            new_state = self.decoder_init_state
            outputs_list = []
            for i in range(self.seq_len):
                self.new_output, new_state = self.decoder_cell(tf.zeros(shape=tf.shape(self.encoder_output_T)[1:]), new_state)  # None
                outputs_list.append(self.new_output)

            self.decoder_outputs = tf.stack(outputs_list, axis=0)  # seq_len * batch_size * hidden_dim
            self.decoder_outputs = tf.transpose(self.decoder_outputs, [1, 0, 2])  # batch_size * seq_len * hidden_dim

            self.output_preds = tf.layers.dense(self.decoder_outputs, units=self.node_num, activation=None)

            return self.input_seqs, self.output_preds, self.encoder_output


    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)

    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars, self.num_network_params, self.num_target_network_params


    def getloss(self, input_seqs):
        feed_dict = {self.target_input_seqs: input_seqs}
        return self.sess.run(self.loss_ce, feed_dict)

    def train(self, input_seqs, num_idx):
        input_seq_new = []
        for seq in input_seqs:
            if 3312 in seq:
                continue
            else:
                input_seq_new.append(seq)
        input_seq_new = np.array(input_seq_new)
        feed_dict = {self.target_input_seqs: input_seq_new}
        return self.sess.run([self.loss_ce, self.optimizer], feed_dict)


    def get_node_embedding(self, input_seqs):
        feed_dict = {self.target_input_seqs: input_seqs}
        node_embedding = self.sess.run(self.encoder_output, feed_dict)
        return np.array(node_embedding)

    def get_batch_embedding(self, dataset):
        user_input = dataset[0]
        num_batch = dataset[2]
        batch_embedding = []
        for batch_index in range(num_batch):
            #batched_user_input = np.array([u for u in user_input[batch_index]])
            node_bag_embedding_list = []
            for node_bag_list in user_input[batch_index]:
                batch_sub_embedding = self.get_node_embedding(np.array(node_bag_list))
                node_bag_embedding_list.append(batch_sub_embedding)
            batch_embedding.append(node_bag_embedding_list)
        return np.array(batch_embedding)

    def get_reward(self, input_seqs):
        feed_dict = {self.target_input_seqs: input_seqs}
        return self.sess.run(self.reward, feed_dict)


    def get_rewards(self, dataset):

        user_input = dataset[0]
        num_batch = dataset[2]
        batch_reward_likelihood = []
        for batch_index in range(num_batch):
            batch_reward_list = []
            for node_bag_list in user_input[batch_index]:
                #print(np.array(node_bag_list).shape)
                #print(np.array(node_bag_list)[0])
                batch_reward = self.get_reward(np.array(node_bag_list))
                #print(batch_reward.shape)
                #print(batch_reward[0])
                batch_reward_list.append(batch_reward)
            batch_reward_likelihood.append(batch_reward_list)

        return np.array(batch_reward_likelihood)


    def get_batch_rewards(self, user_input):

        batch_reward_list = []
        for node_bag_list in user_input:
            #print(np.array(node_bag_list).shape)
            #print(np.array(node_bag_list)[0])
            batch_reward = np.array(self.get_reward(np.array(node_bag_list)))
            is_loglikelihood = np.isnan(batch_reward)
            batch_reward[is_loglikelihood] = 0
            #print(batch_reward)
            #print(batch_reward.shape)
            #print(batch_reward[0])
            batch_reward_list.append(batch_reward)

        return np.array(batch_reward_list)


    
            
