class setting():

    def __init__(self):
        ############ hyper-parameters for stne ###########
        self.h_dim = 500
        self.dropout = 0.3 
        self.lr = 0.01 
        self.depth = 1
        self.tau = 0.3
        #self.num_items = 5
        self.seq_len = 10
        self.stne_epochs = 1
        self.hidden_dim=500
        self.trainable=False


        ############# hyper-parameters for agent ################
        self.agent_epochs = 2
        self.agent_pretrain_lr = 0.001
        self.agent_pretrain_tau = 0.01
        self.agent_lr = 0.005
        self.agent_tau = 0.005
        self.high_state_size = 1000
        self.low_state_size = 1000
        self.agent_weight_size = 100
        self.sample_cnt = 10


        ############# hyper-parameres about the dataset ##############

        self.folder = './data/citeseer/'
        self.data_name = 'citeseer'
        self.batch_size = 8 #256
        self.agent_pretrain = True
        self.stne_pretrain = True
        self.pre_agent = "./checkpoint/citeseer-pre-agent/"
        self.pre_stne = "./checkpoint/citeseer-pre-stne/"
        self.agent = "./checkpoint/citeseer-agent/"
        self.stne = "./checkpoint/citeseer-stne/"
        self.agent_verbose = 1
        self.stne_verbose = 3
