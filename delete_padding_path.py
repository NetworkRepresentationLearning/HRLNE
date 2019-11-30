
import os
import sys
sys.path.append("./")
sys.path.append("../")

import codecs

import numpy as np 


fw = codecs.open("selected_path_citeseer_1_2_clear.txt", "w", "utf-8")

seq_list = []
for line in codecs.open("selected_path_citeseer_1_2.txt", "r", "utf-8"):
    token = line.strip("\r\n").split(" ")
    if int(token[0]) == 3312:
        continue
    else:
        seq_list.append(line.strip("\r\n"))
        #fw.write(line)

shuffled_batch_indexes = np.random.permutation(len(seq_list))
for index in shuffled_batch_indexes:
    fw.write(seq_list[index] + "\n")
        
fw.close()
