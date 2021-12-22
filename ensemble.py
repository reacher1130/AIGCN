import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets

# label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label = open('/mnt/DataDrive246/gaofeng/Datasets/ntu120_2people/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
# r1 = open('./work_dir/' + dataset + '/test_joint/epoch1_test_score.pkl', 'rb')
# # r1 = open('/home/ubuntu/users/gaofeng/Action_Recognition_Code/DecoupleGCN/work_dir/ntu_joint_xview/eval_results/best_acc.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
# r2 = open('./work_dir/' + dataset + '/test_bone/epoch1_test_score.pkl', 'rb')
# r2 = list(pickle.load(r2).items())
r1 = open(r'/mnt/DataDrive246/gaofeng/Action Recognition Code/two_man_interaction/work_dir/fusion/epoch1_test_score_joint.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(r'/mnt/DataDrive246/gaofeng/Action Recognition Code/two_man_interaction/work_dir/fusion/epoch1_test_score_bone.pkl', 'rb')
r2 = list(pickle.load(r2).items())

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
