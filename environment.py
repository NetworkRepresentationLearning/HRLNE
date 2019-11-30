import numpy as np

class Environment():
    def __init__(self):
        self.gamma = 0.5

    def initilize_state(self, stne, traindata, high_state_size, low_state_size, seq_len):
        self.high_state_size = high_state_size
        self.low_state_size = low_state_size
        self.seq_len = seq_len
        self.padding_number = 3312
        self.batch_embedding = stne.get_batch_embedding(traindata)
        self.origin_train_rewards = stne.get_rewards(traindata)
        #print("origin_train_rewards shape", np.array(self.origin_train_rewards[0]).shape)
        #self.origin_test_rewards = stne.get_rewards(testdata)
        self.embedding_size = len(self.batch_embedding[0][0][0][0])  # 16
        #print("embedding_size:", self.embedding_size)
        self.set_train_original_rewards()

    def set_train_original_rewards(self):
        self.origin_rewards = self.origin_train_rewards



    
    def reset_state(self, input_seq, num_idx, batch_size, max_seq_num, batch_index):
        
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.batch_index = batch_index
        self.max_seq_num = max_seq_num
        self.num_idx = num_idx

        self.origin_prob = np.zeros((self.batch_size, 1), dtype=np.float32)

        #self.current_vector = np.zeros((self.batch_size, self.seq_len*self.embedding_size), dtype=np.float32)
        
        self.vector_mean_origin = np.zeros((self.batch_size, self.embedding_size),  dtype=np.float32)
        self.vector_sum = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_mean = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)

        self.num_selected = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix = np.zeros((self.batch_size, self.max_seq_num), dtype=np.int)
        self.state_matrix = np.zeros((self.batch_size, self.max_seq_num, self.low_state_size), dtype=np.float32)
        self.selected_input = np.full((self.batch_size, self.max_seq_num), self.seq_len)


    def get_overall_state(self):

        origin_prob = np.array(self.origin_rewards[self.batch_index]) #(batch_size, 1)
        #print("origin_rewards shape ", np.array(self.origin_rewards[self.batch_index]).shape)
        #print("origin_prob shape ", origin_prob.shape)
        self.batch_embedding_current = self.batch_embedding[self.batch_index]
        #print(np.sum(self.batch_embedding_current, axis=1).shape)
        vector_mean_origin_1 = np.mean(self.batch_embedding_current, axis=1)
        vector_mean_origin_2 = np.reshape(np.mean(vector_mean_origin_1, axis=1), (self.batch_size, self.embedding_size))
        self.vector_mean_origin = vector_mean_origin_2
        #print("vector_mean_origin shape %s", self.vector_mean_origin.shape)

        #return np.concatenate((self.vector_mean_origin, origin_prob),1)
        return self.vector_mean_origin

   
    def get_state(self, step_index):
        self.origin_prob = np.array(self.origin_rewards[self.batch_index])  # (batch_size, 1)
        vector_current_1 = np.mean(np.array(self.batch_embedding[self.batch_index])[:, step_index], axis=1)
        self.vector_current = np.reshape(vector_current_1, (self.batch_size, self.embedding_size))

        #return np.concatenate((self.vector_mean, self.vector_current), 1)
        return self.vector_current



    def update_state(self, low_action, low_state, step_index):
        self.action_matrix[:, step_index] = low_action
        self.state_matrix[:, step_index] = low_state

        self.num_selected = self.num_selected + low_action
        self.vector_sum = self.vector_sum + np.multiply(np.reshape(low_action, (-1, 1)), self.vector_current)
        num_selected_array = np.reshape(self.num_selected, (-1, 1))
        self.vector_mean = np.where(num_selected_array != 0, self.vector_sum / num_selected_array, self.vector_sum)


    def get_action_matrix(self):
        return self.action_matrix

    def get_state_matrix(self):
        return self.state_matrix

    def get_selected_paths(self, high_action):
        notrevised_index = []
        revised_index = []
        delete_index = []
        keep_index = []
        selected_input_seq = np.zeros((self.batch_size, self.max_seq_num, self.seq_len), dtype=np.int)
        for index in range(self.batch_size):

            selected = []
            for path_index in range(self.max_seq_num):
                if self.action_matrix[index, path_index] == 1:
                    selected.append(list(self.input_seq[index, path_index]))
            #print("select num ", len(selected))


            # revise
            if high_action[index] == 1:
                # delete
                if len(selected) == 0:
                    delete_index.append(index)
                # keep
                if len(selected) == self.num_idx[index]:
                    keep_index.append(index)
                revised_index.append(index)
            # not revise
            if high_action[index] == 0:
                notrevised_index.append(index)

            # random select one course from the original enrolled courses if no course is selected by the agent, change the number of selected courses as 1 at the same time
            if len(selected) == 0:
                #original_path_set = list(self.input_seq[index])
                original_path_set = list([sub_index for sub_index in range(self.max_seq_num)])
                # if self.padding_number in original_path_set:
                #     original_path_set.remove(self.padding_number)
                random_path_index = np.random.choice(original_path_set, 1)[0]
                random_path = self.input_seq[index, random_path_index]
                #print("random_path", random_path)
                selected.append(random_path)
                self.num_selected[index] = 1

            for path_index in range(self.max_seq_num - len(selected)):
                selected.append([self.padding_number for i in range(0, self.seq_len)])
            #print(np.array(selected))
            #print(np.array(selected).shape)
            selected_input_seq[index, :, :] = np.reshape(np.array(selected), (self.max_seq_num, self.seq_len))
        
        
        nochanged = notrevised_index + keep_index
        selected_input_seq[nochanged] = self.input_seq[nochanged][:,:]
        self.num_selected[nochanged] = np.reshape(self.num_idx[nochanged],(-1,))
        return selected_input_seq, self.num_selected, notrevised_index, revised_index, delete_index, keep_index


    def get_reward(self, stne, batch_index, high_actions, selected_seq_input, batched_num_idx):
        batch_size = selected_seq_input.shape[0]
        #print(selected_seq_input.shape)

        # difference between likelihood
        loglikelihood = np.array(stne.get_batch_rewards(selected_seq_input))
        # is_loglikelihood = np.isnan(loglikelihood)
        # loglikelihood[is_loglikelihood] = 0

        loglikelihood_ave = np.sum(loglikelihood, axis=1)
        loglikelihood_ave = loglikelihood_ave / self.num_selected
        #print("loglikelihood_ave", loglikelihood_ave)
        # for sub_loglikelihood in loglikelihood_ave:
        #     print(sub_loglikelihood)
        old_likelihood = self.origin_rewards[batch_index]
        old_likelihood_ave = np.mean(old_likelihood, axis=1)
        likelihood_diff = np.array(loglikelihood_ave-old_likelihood_ave)
        #print("likelihood_diff", likelihood_diff)
        #likelihood_diff = np.array(loglikelihood_ave)
        likelihood_diff = np.where(high_actions == 1, likelihood_diff, np.zeros(batch_size))

        reward1 = likelihood_diff

        return reward1
