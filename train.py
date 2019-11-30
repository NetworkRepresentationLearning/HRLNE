import numpy as np
import tensorflow as tf
import os
from time import time
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from environment import Environment
from agent import AgentNetwork
from stne import STNE
from DataParser import Dataset
# from Evaluation import eval_rating
from Setting import setting
from sklearn.linear_model import LogisticRegression
from classify import Classifier

import warnings
warnings.filterwarnings('ignore')

global padding_number
global env

import traceback


def _get_high_action(prob, Random):
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where(random_number < prob, np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))
    else:
        return np.where(prob >= 0.5, np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))

def _get_low_action(prob, user_input_column, Random):
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where(random_number < prob , np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))
    else:
        return np.where(prob >= 0.5, np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))

def _get_low_action_old(prob, user_input_column, padding_number, Random):
    user_input_column_new = []
    for path in user_input_column:
        if np.array(path).all() and np.array(path)[0] == padding_number:
            user_input_column_new.append(padding_number)
        else:
            user_input_column_new.append(padding_number-1)
    user_input_column_new = np.array(user_input_column_new)
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where((random_number < prob) & (user_input_column_new != padding_number), np.ones(batch_size,dtype=np.int),
                        np.zeros(batch_size,dtype=np.int))
    else:
        return np.where((prob >= 0.5) & (user_input_column_new != padding_number), np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))


def sampling_RL(seq_input, num_idx, batch_index, agent, Random=True):

    batch_size = seq_input.shape[0]
    max_seq_num = seq_input.shape[1]
    env.reset_state(seq_input, num_idx, batch_size, max_seq_num, batch_index)
    high_state = env.get_overall_state()
    # for sub_high_state in high_state:
    #     print(sub_high_state)
    high_prob = agent.predict_high_target(high_state)
    # print("high_prob ", high_prob)
    high_action = _get_high_action(high_prob, Random)
    print("high_action ", high_action)
    # if np.sum(np.array(high_action)) <=1 or np.sum(np.array(high_action)) >= 24:
    #     print(high_state[0])
    #     print(high_state[1])
    #     print(high_state[2])
    #     print(high_state[3])
    #     print(high_state[4])

    for i in range(max_seq_num):
        low_state = env.get_state(i)
        #print(low_state[0])
        low_prob = agent.predict_low_target(low_state)
        low_action = _get_low_action_old(low_prob, seq_input[:, i], padding_number, Random)
        #print(low_prob, low_action)
        env.update_state(low_action, low_state, i)
    select_seq_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index = env.get_selected_paths(high_action)

    return high_action, high_state, select_seq_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index


def train(trained_set, sess, agent, stne, trainset, args, stne_trainable=True, agent_trainable=True):
    train_seq_input, train_num_idx, train_batch_num  = (trainset[0], trainset[1], trainset[2])
    sample_times = args.sample_cnt
    high_state_size = args.high_state_size
    low_state_size = args.low_state_size
    avg_loss = 0

    shuffled_batch_indexes = np.random.permutation(int(train_batch_num))
    for key_inde, batch_index in enumerate(shuffled_batch_indexes):

        batched_seq_input_origin = np.array([u for u in train_seq_input[batch_index]])
        #print(batched_seq_input_origin.shape)

        batch_size = batched_seq_input_origin.shape[0]
        max_seq_num = batched_seq_input_origin.shape[1]

        batched_seq_input = np.reshape(batched_seq_input_origin, (batch_size*max_seq_num,10))
        
        batched_num_idx = np.reshape(train_num_idx[batch_index], (-1,1))

        #batched_seq_input_rl = np.reshape(batched_seq_input_origin, ())

        

        train_begin = time()
        train_loss = 0
        agent.assign_active_high_network()
        agent.assign_active_low_network()
        stne.assign_active_network()
        if agent_trainable:

            sampled_high_states = np.zeros((sample_times, batch_size, high_state_size), dtype=np.float32)
            sampled_high_actions = np.zeros((sample_times, batch_size), dtype=np.int)

            sampled_low_states = np.zeros((sample_times, batch_size, max_seq_num, low_state_size), dtype=np.float32)
            sampled_low_actions = np.zeros((sample_times, batch_size, max_seq_num), dtype=np.float32)

            sampled_high_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)
            sampled_low_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)

            sampled_revise_index = []

            avg_high_reward = np.zeros((batch_size), dtype=np.float32)
            avg_low_reward = np.zeros((batch_size), dtype=np.float32)

            for sample_time in range(sample_times):
                #print("sample_time %d", sample_time)
                high_action, high_state, select_seq_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index =  \
                    sampling_RL(batched_seq_input_origin, batched_num_idx, batch_index, agent)
                sampled_high_actions[sample_time, :] = high_action
                sampled_high_states[sample_time, :] = high_state
                sampled_revise_index.append(revised_index)


                reward = env.get_reward(stne, batch_index, high_action, select_seq_input, select_num_idx)
                #print("key_index", key_inde+1 , "reward ", reward)

                # reward = np.sqrt(np.multiply(reward,env.num_selected)/env.num_idx[batch_index])   # Geometric mean
                avg_high_reward += reward
                avg_low_reward += reward
                sampled_high_rewards[sample_time, :] = reward
                sampled_low_rewards[sample_time, :] = reward
                sampled_low_actions[sample_time, :] = env.get_action_matrix()
                sampled_low_states[sample_time, :] = env.get_state_matrix()

            avg_high_reward = avg_high_reward / sample_times
            avg_low_reward = avg_low_reward / sample_times
            high_gradbuffer = agent.init_high_gradbuffer()
            low_gradbuffer = agent.init_low_gradbuffer()
            for sample_time in range(sample_times):
                high_reward = np.subtract(sampled_high_rewards[sample_time], avg_high_reward)
                high_gradient = agent.get_high_gradient(sampled_high_states[sample_time], high_reward,sampled_high_actions[sample_time] )
                agent.train_high(high_gradbuffer, high_gradient)
                #print(sess.run(agent.high_network_params))
                # if sample_time == 5:
                #     parameters = agent.get_network_parameter()
                #     print(("Activate/high/Weights_for_MLP"))
                #     print(parameters[0])
                #     print(("Activate/high/Bias_for_MLP"))
                #     print(parameters[1])
                #     print(("Activate/high/H_for_MLP"))
                #     print(parameters[2])

                revised_index = sampled_revise_index[sample_time]
                low_reward = np.subtract(sampled_low_rewards[sample_time], avg_low_reward)
                low_reward_row = np.tile(np.reshape(low_reward[revised_index], (-1, 1)), max_seq_num)
                low_gradient = agent.get_low_gradient(
                    np.reshape(sampled_low_states[sample_time][revised_index], (-1, low_state_size)),
                    np.reshape(low_reward_row, (-1,)),
                    np.reshape(sampled_low_actions[sample_time][revised_index], (-1,)))
                agent.train_low(low_gradbuffer, low_gradient)

            if stne_trainable:
                _, _, select_seq_input, select_num_idx, _, _, _, _ =  \
                    sampling_RL(batched_seq_input_origin, batched_num_idx, batch_index, agent, Random=False)
                train_loss,_ = stne.train(np.reshape(select_seq_input, (batch_size*max_seq_num,10)), np.reshape(select_num_idx,(-1,1)))
                select_seq_input_selected = np.reshape(select_seq_input, (batch_size*max_seq_num,10))
                all_trained = check_all_node_trained(trained_set, select_seq_input_selected, stne.node_num)
                if all_trained and (key_inde+1) % 5 == 0:
                    try:
                        f1_micro_list = node_classification(stne, trainset, stne.seq_len, stne.node_num, dataset.X, dataset.Y)
                        print("key_index", key_inde+1 , "f1_list",f1_micro_list)
                    except:
                        print("node classification FAILED")
                        erroStack = traceback.format_exc()
                        print(erroStack)

        else:
            train_loss,_ = stne.train(batched_seq_input, batched_num_idx)
            all_trained = check_all_node_trained(trained_set, batched_seq_input, stne.node_num)
            if all_trained and (key_inde+1) % 5 == 0:
                try:
                    f1_micro_list = node_classification(stne, trainset, stne.seq_len, stne.node_num, dataset.X, dataset.Y)
                    print("key_index", key_inde+1 , "f1_list",f1_micro_list)
                except:
                    print("node classification FAILED")
                    erroStack = traceback.format_exc()
                    print(erroStack)

        avg_loss += train_loss
        # if stne_trainable and (key_inde+1) % 20 == 0:
        #     print("avg_loss is %f", avg_loss/(key_inde+1))
        train_time = time() - train_begin


        # Update parameters
        if agent_trainable:
            agent.update_target_high_network()
            agent.update_target_low_network()
            if stne_trainable:
                stne.update_target_network()
        else:
            stne.assign_target_network()

    return avg_loss / train_batch_num


def get_avg_reward(agent, trainset):
    train_seq_input, train_num_idx, train_batch_num  = (trainset[0], trainset[1], trainset[2])
    avg_reward, total_selected_paths, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances = 0,0,0,0,0,0
    total_instances = 0
    test_begin = time()
    for batch_index in range(train_batch_num):
        batched_seq_input = np.array([u for u in train_seq_input[batch_index]])
        batched_num_idx = np.reshape(train_num_idx[batch_index], (-1,1))

        high_action, high_state, select_seq_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index =  \
        sampling_RL(batched_seq_input, batched_num_idx, batch_index, agent, Random=False)
        reward = env.get_reward(stne_model, batch_index, high_action, select_seq_input, select_num_idx)

        avg_reward += np.sum(reward)
        total_selected_paths += np.sum(select_num_idx)
        total_revised_instances += len(revised_index)
        total_notrevised_instances += len(notrevised_index)
        total_deleted_instances += len(delete_index)
        total_keep_instances += len(keep_index)
        total_instances += batched_seq_input.shape[0]
    test_time = time() - test_begin
    avg_reward = avg_reward / total_instances
    return avg_reward, total_selected_paths, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances, test_time



def reduce_seq2seq_hidden_add(sum_dict, count_dict, batched_seq_input, batch_enc, seq_len):
    for i_seq in range(len(batched_seq_input)):
        for j_node in range(seq_len):
            nid = batched_seq_input[i_seq, j_node]
            if nid in sum_dict:
                sum_dict[nid] = sum_dict[nid] + batch_enc[i_seq, j_node, :]
            else:
                sum_dict[nid] = batch_enc[i_seq, j_node, :]
            if nid in count_dict:
                count_dict[nid] = count_dict[nid] + 1
            else:
                count_dict[nid] = 1
    return sum_dict, count_dict


def reduce_seq2seq_hidden_avg(sum_dict, count_dict, node_num):
    vectors = []
    for nid in range(node_num):
        vectors.append(sum_dict[nid] / count_dict[nid])
    return np.array(vectors)


def node_classification(seqne, trainset, seq_len, node_n, samp_idx, label):
    enc_sum_dict = {}
    node_cnt = {}
    train_seq_input, train_num_idx, train_batch_num  = (trainset[0], trainset[1], trainset[2])
    shuffled_batch_indexes = np.random.permutation(int(train_batch_num))
    for key_inde, batch_index in enumerate(shuffled_batch_indexes):
        batched_seq_input_origin = np.array([u for u in train_seq_input[batch_index]])

        batch_size = batched_seq_input_origin.shape[0]
        max_seq_num = batched_seq_input_origin.shape[1]

        batched_seq_input = np.reshape(batched_seq_input_origin, (batch_size*max_seq_num,10))
        batch_enc = seqne.get_node_embedding(batched_seq_input)
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, batched_seq_input,
                                                           batch_enc.astype('float32'), seq_len)
    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    f1_micro_list = []
    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
    for ratio in clf_ratio:
        lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
        f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
        f1_micro_list.append(f1_micro)
    return f1_micro_list

def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        return False

def print_agent_message(epoch, avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances,test_time, train_time ):
    partial_revised = total_revised_instances-total_deleted_instances-total_keep_instances
    logging.info(
        "Epoch %d : avg reward = %.4f, courses (keep = %d), instances (revise = %d, notrevise = %d, delete = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
            epoch,  avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))
    print(
        "Epoch %d : avg reward = %.4f, courses (keep = %d), instances (revise = %d, notrevise = %d, delete = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
            epoch,  avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                filename='log.txt',
                filemode='w',
                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    args = setting()
    config = tf.ConfigProto()
    dataset = Dataset(args.folder, args.data_name, args.batch_size)
    padding_number = 3312

    node_feature = dataset.node_fea
    X, Y = dataset.X, dataset.Y
    node_num = dataset.node_num
    fea_dim = dataset.fea_dim
    node_seq = dataset.node_seq
    node_bag_seq = dataset.node_bag_seq_list
    node_list = dataset.node_list
    train_data = dataset._preprocess()


    env = Environment()

    with tf.Session(config=config) as sess:
        stne_model = STNE(sess, args, node_fea=node_feature)
        agent = AgentNetwork(sess, args)

        pre_stne_saver = tf.train.Saver()
        pre_agent_saver = tf.train.Saver()
        stne_saver = tf.train.Saver()
        agent_saver = tf.train.Saver()

        if args.stne_pretrain:
            node_classification_result = [0.60, 0.60, 0.60, 0.60, 0.60]
            sess.run(tf.global_variables_initializer())

            trained_set = set()

            for epoch in range(args.stne_epochs):
                print("stne epoch ", epoch, "start training")
                train_begin = time()
                
                train_loss = train(trained_set, sess, agent, stne_model, train_data, args, stne_trainable=True, agent_trainable=False)
                
                train_time = time() - train_begin
                stne_model.assign_target_network()
                train_data = dataset._preprocess()
                # recommender.assign_target_network()
                if epoch % 1 == 0:
                    test_begin = time()
                    ratio_flag = 0
                    try:
                        f1_micro_list = node_classification(stne_model, train_data, stne_model.seq_len, stne_model.node_num, dataset.X, dataset.Y)
                        test_time = time() - test_begin
                        print("Epoch", epoch, f1_micro_list, test_time, train_loss, train_time)
                        for ratio_index in range(0, len(f1_micro_list)):

                            if f1_micro_list[ratio_index] >= node_classification_result[ratio_index]:
                                node_classification_result[ratio_index] = f1_micro_list[ratio_index]
                                ratio_flag += 1
                    except:
                        print("node classification FAILED")
                        erroStack = traceback.format_exc()
                        print(erroStack)

                    if ratio_flag >= 1:
                        pre_stne_saver.save(sess, args.pre_stne, global_step=epoch)
            print ("stne pretrain OK")
            logging.info("stne pretrain OK")
        print ("Load stne pre-trained stne from ", args.pre_stne)
        logging.info("Load best pre-trained stne from %s " % args.pre_stne)
        pre_stne_saver.restore(sess, tf.train.get_checkpoint_state(os.path.dirname(args.pre_stne+'checkpoint')).model_checkpoint_path)
        print ("Evaluate pre-trained stne based on original test instances")
        logging.info("Evaluate pre-trained stne based on original test instances")
        test_begin = time()
        test_time = time() - test_begin


        #Agent pretrain
        env.initilize_state(stne_model, train_data, args.high_state_size, args.low_state_size, args.seq_len)
        if args.agent_pretrain:
            best_avg_reward = -100000

            trained_set = set()
            for epoch in range(args.agent_epochs):
                print("agent epoch ", epoch, "start training")
                train_begin = time()
                train_loss = train(trained_set, sess, agent, stne_model, train_data, args, stne_trainable=False, agent_trainable=True)
                train_time = time() - train_begin
                if epoch % args.agent_verbose == 0:
                    test_begin = time()
                    avg_reward, total_selected_paths, total_revised_instances, total_notrevised_instances, \
                    total_deleted_instances, total_keep_instances, test_time = get_avg_reward(agent, train_data)
                    print("avg_reward ", avg_reward)

                    test_time = time() - test_begin
                    print_agent_message(epoch, avg_reward, total_selected_paths, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances,test_time,train_time)
                    if avg_reward >= best_avg_reward:
                        best_avg_reward = avg_reward
                        pre_agent_saver.save(sess, args.pre_agent, global_step=epoch)
            print ("Agent pretrain OK")
        print ("Load best pre-trained agent from", args.pre_agent)
        logging.info("Load best pre-trained agent from %s" % args.pre_agent)
        pre_agent_saver.restore(sess, tf.train.get_checkpoint_state(os.path.dirname(args.pre_agent+'checkpoint')).model_checkpoint_path)




        #Agent and stne jointly train
        print ("Begin to jointly train")
        logging.info("Begin to jointly train")
        node_classification_result = [0.60, 0.60, 0.60, 0.60, 0.60]
        best_avg_reward = -100000
        print (agent.tau, agent.lr)
        agent.udpate_tau(args.agent_tau)
        agent.update_lr(args.agent_lr)
        print (agent.tau, agent.lr)
        for epoch in range(0, 5):
            trained_set = set()

            train_begin = time()
            train_loss = train(trained_set, sess, agent, stne_model, train_data, args, stne_trainable=True, agent_trainable=True)
            train_time = time() - train_begin
            if epoch % 1 == 0:

                test_begin = time()
                avg_reward, total_selected_paths, total_revised_instances,total_notrevised_instances,\
                 total_deleted_instances,total_keep_instances, test_time = get_avg_reward(agent, train_data)
                test_time = time() - test_begin
                print_agent_message(epoch, avg_reward, total_selected_paths, total_revised_instances, \
                    total_notrevised_instances, total_deleted_instances,total_keep_instances,test_time,train_time)
                if avg_reward >= best_avg_reward:
                    best_avg_reward = avg_reward
                    agent_saver.save(sess, args.agent, global_step=epoch)

            if epoch % 1 == 0:
                test_begin = time()
                ratio_flag = 0
                try:
                    f1_micro_list = node_classification(stne_model, train_data, stne_model.seq_len, stne_model.node_num, dataset.X, dataset.Y)
                    test_time = time() - test_begin
                    print("Epoch", epoch, f1_micro_list, test_time, train_loss, train_time)
                    
                    for ratio_index in range(0, len(f1_micro_list)):

                        if f1_micro_list[ratio_index] >= node_classification_result[ratio_index]:
                            node_classification_result[ratio_index] = f1_micro_list[ratio_index]
                            ratio_flag += 1
                except:
                    print("node classification FAILED")
                    erroStack = traceback.format_exc()
                    print(erroStack)
                if ratio_flag >= 1:
                    stne_saver.save(sess, args.stne, global_step=epoch)
            # To update the embeddings and the original rewards or not?
            env.initilize_state(stne_model, train_data, args.high_state_size, args.low_state_size, args.seq_len)
        print ("Jointly train OK")
        logging.info("Jointly train OK")
        print ("Evaluate jointly-trained recommender based on the selected test instances by the jointly-trained agent")
        logging.info("Evaluate jointly-trained recommender based on the selected test instances by the jointly-trained agent")
        test_begin = time()
        try:
            f1_micro_list = node_classification(stne_model, train_data, stne_model.seq_len, stne_model.node_num, dataset.X, dataset.Y)
            test_time = time() - test_begin
            print("Epoch", epoch, f1_micro_list, test_time)
        except:
            print("node classification FAILED")
            erroStack = traceback.format_exc()
            print(erroStack)


