import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import argparse
import torch
import os
import random
import zipfile
from tqdm import tqdm
import pickle
import shutil
from mpl_toolkits import mplot3d

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def unzip(sourceFile, targetPath):
    file = zipfile.ZipFile(sourceFile, 'r')
    file.extractall(targetPath)
    print('success to unzip file')


def generate_data(target, normalized=False):
    total = []
    labels = []

    for filename in os.listdir(target):
        if filename == 'train.json' or filename == 'val.json':
            continue
        # filename = '%02d' % i
        mainFile = os.path.join(target, filename)
        # print(mainFile)
        if os.listdir(os.path.join(target, filename))[0] == '.DS_Store':
            os.remove(os.path.join(os.path.join(target, filename), '.DS_Store'))
        # subFile = os.listdir(mainFile)[0]
        # print(subFile)

        sub_path = os.path.join(mainFile)
        cats = sorted(os.listdir(sub_path))
        # print(cat)
        for cat in cats:
            label = int(cat) - 1
            # print(label)
            cat_path = os.path.join(sub_path, cat)
            nums = sorted(os.listdir(cat_path))
            if nums[0] == '.DS_Store':
                nums = nums[1::]
            for num in nums:
                list_1 = []
                list_2 = []
                one = []
                num_path = os.path.join(cat_path, num)
                list = os.listdir(num_path)
                for i in range(len(list)):
                    if list[i] == 'skeleton_pos.txt':
                        continue
                    delfile = os.path.join(num_path, list[i])
                    os.remove(delfile)
                txt_path = os.path.join(num_path, "skeleton_pos.txt")

                # print(txt_path)
                with open(txt_path) as f:
                    data = f.readlines()
                for row in data:
                    posture = row
                    pose_1 = []
                    pose_2 = []
                    posture_data = [x.strip() for x in posture.split(',')]

                    joint_info = {}
                    if normalized:
                        for i, n in enumerate(range(1, len(posture_data), 3)):
                            joint_info[i + 1] = [float(posture_data[n]),
                                                 float(posture_data[n + 1]),
                                                 float(posture_data[n + 2])]
                    else:
                        for i, n in enumerate(range(1, len(posture_data), 3)):
                            joint_info[i + 1] = [(1280 - float(posture_data[n]) * 2560),
                                                 (960 - (float(posture_data[n + 1])) * 1920),
                                                 ((float(posture_data[n + 2]) * 10000) / 7.8125)]
                    person_1 = {k: joint_info[k] for k in range(1, 16, 1)}
                    person_2 = {k - 15: joint_info[k] for k in range(16, 31, 1)}
                    # print(person_1)

                    for key, value in person_1.items():
                        pose_1.append(value[0:3])
                    array = np.array(pose_1)
                    list_1.append(array)

                    for key, value in person_2.items():
                        pose_2.append(value[0:3])
                    array = np.array(pose_2)
                    list_2.append(array)

                list_1 = np.array(list_1)
                list_2 = np.array(list_2)
                one.append(list_1)
                one.append(list_2)
                labels.append(label)
                total.append(one)
    return total, labels


def interpolate_data(total, labels):
    count = 0
    interpolate = []
    for i in range(len(labels)):
        slice = np.array(total[i])
        if slice.shape[1] > 35:
            count += 1
        temp = []
        for person in range(2):
            pose = slice[person]
            tensor_1 = torch.tensor(pose)
            tensor_1 = tensor_1.permute(1, 2, 0)
            target_1 = torch.nn.functional.interpolate(tensor_1, size=300, mode='linear')
            # print(target_1.shape)
            target_1 = target_1.permute(2, 0, 1)
            temp.append(target_1.numpy())
        interpolate.append(np.array(temp))
    interpolate = np.array(interpolate)
    print(interpolate.shape)
    return interpolate, labels


def show_demo(interpolate):
    target_1 = interpolate[0, 0]
    target_2 = interpolate[0, 1]
    connect_map = [[1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 7, 8, 10, 11, 13, 14],
                   [2, 3, 4, 7, 4, 7, 10, 13, 5, 6, 8, 9, 11, 12, 14, 15]]
    connect_map = np.array(connect_map) - 1
    connect_map = connect_map.tolist()
    figure = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.xlim(-1, 1)
    ax = Axes3D(figure)
    # plt.ylim(-1, 1)

    image = []
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='ME'), bitrate=1800)
    for i in range(300):
        p = []
        pose = target_1[i]  # 15,3
        for j in range(15):
            value = pose[j]
            # print(value)
            p += ax.plot(value[0], value[1], value[2], 'bo')
            # plt.annotate(joint_details[key], (value[0], value[1]))
        for m, n in zip(connect_map[0], connect_map[1]):
            p += ax.plot((pose[m, 0], pose[n, 0]), (pose[m, 1], pose[n, 1]), (pose[m, 2], pose[n, 2]), 'b--')
        pose = target_2[i]
        for j in range(15):
            value = pose[j]
            # print(value)
            p += ax.plot(value[0], value[1], value[2], 'go')
            # plt.annotate(joint_details[key], (value[0], value[1]))
        for m, n in zip(connect_map[0], connect_map[1]):
            p += ax.plot((pose[m, 0], pose[n, 0]), (pose[m, 1], pose[n, 1]), (pose[m, 2], pose[n, 2]), 'g--')
        ax.set_title(i)
        ax.view_init(80, -45)
        # plt.show()
        image.append(p)

    image_animation = animation.ArtistAnimation(figure, image, interval=10, repeat=False)
    image_animation.save('image.gif', fps=10)


def split_train_val(interpolate, labels, k, target, split_val=False):
    aa = []
    bb = []
    zz = []
    ACTIONS = ['Approaching', 'Departing', 'Kicking', 'Punching', 'Pushing', 'Hugging',
               'ShakingHands', 'Exchanging']
    Action_label = []
    size = len(labels)
    for i, s in enumerate(labels):
        Action_label.append(ACTIONS[s])
    cc = list(zip(interpolate, labels, Action_label))
    random.shuffle(cc)
    aa[:], bb[:], zz[:] = zip(*cc)
    a = []
    b = []
    z = []
    for i in range(k):
        a.append(aa[i*108: (i+1)*108])
        b.append(bb[i*108: (i+1)*108])
        z.append(zz[i*108: (i+1)*108])

    pailie = [[1,2,3,4,0],
              [0,1,2,3,4],
              [0,1,2,4,3],
              [0,1,3,4,2],
              [0,2,3,4,1]]
    for i, p in enumerate(pailie):
        temp_a = []
        temp_b = []
        temp_z = []
        train_dict = {}
        for j, _ in enumerate(p):
            if j == 4:
                break
            temp_a += a[p[j]]
            temp_b += b[p[j]]
            temp_z += z[p[j]]

        if split_val:
            train_x = np.array(temp_a)
            train_y = np.array(temp_b)
            train_z = temp_z

        print(train_x.shape)
        print(train_y.shape)
        train_dict["x"] = train_x
        train_dict["label"] = train_y
        train_dict["action_label"] = train_z
        save_to_json(train_dict, "{}/{}_train.json".format(target, i))
        if split_val:
            valid_dict = {}
            valid_x = np.array(a[p[-1]])
            valid_y = np.array(b[p[-1]])
            valid_z = z[p[-1]]
            print(valid_x.shape)
            print(valid_y.shape)
            valid_dict["x"] = valid_x
            valid_dict["label"] = valid_y
            valid_dict["action_label"] = valid_z
            save_to_json(valid_dict, "{}/{}_val.json".format(target, i))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_to_json(dic, target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)
    file = open(target_dir, 'w')
    json.dump(dumped, file)
    file.close()


def read_from_json(target_dir):
    f = open(target_dir, 'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data


# target = "./../data/SBU/"
# data, labels = generate_data(target, normalized=True)
#
# interpolate = interpolate_data(data, labels)
# # show_demo(interpolate)
# split_train_val(interpolate, labels, target)


# source = 'D:\Action Recognition Code\SBU_NOISY'
# target = './../data/SBU_NOISY'
# for filename in os.listdir(source):
#     filepath = os.path.join(source, filename)
#     unzip(filepath, target)


def gendata(p, data_path, data_out_path, label_out_path):
    # target = "./../data/SBU/"

    max_frame = 300
    num_joint = 15
    num_person = 2

    part_dir = os.path.join(data_path, '{}.json'.format(p))
    dict = read_from_json(part_dir)
    dict_data = np.array(dict["x"]).transpose(0, 4, 2, 3, 1)  # n,m,t,v,c
    sample_name = dict["action_label"]
    dict_label = dict["label"]
    sample_label = []
    fp = np.zeros((len(dict_label), 3, max_frame, num_joint, num_person), dtype=np.float32)  # N,C,T,V,M
    for i, s in enumerate(tqdm(dict_label)):
        data, label = dict_data[i], dict_label[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SBU-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='./../data/')
    parser.add_argument(
        '--out_folder', default='./../data/processed_SBU')
    parser.add_argument(
        '--split_val', default=True)
    arg = parser.parse_args()
    dir_list = os.listdir(arg.data_path)
    origin_data = []
    origin_label = []
    interpolate = []
    for dir in dir_list:
        sub_path = os.path.join(arg.data_path, dir)
        if dir == 'processed_SBU':
            shutil.rmtree(sub_path)
            continue
        if dir.endswith('.json'):
            os.remove(os.path.join(sub_path))
            continue

        json_data, json_labels = generate_data(sub_path, normalized=True)
        origin_label += json_labels
        origin_data += json_data
    interpolate, labels = interpolate_data(origin_data, origin_label)
    show_demo(interpolate)
    # split_train_val(interpolate, labels, 5, arg.data_path, arg.split_val)
    # if arg.split_val:
    #     part = ['val', 'train']
    # else:
    #     part = ['train']
    # for p in part:
    #     print('SBU Dataset:', p)
    #     # data_path = '{}/{}.json'.format(arg.data_path, p)
    #     # label_path = '{}/{}.json'.format(arg.data_path, p)
    #     if not os.path.exists(arg.out_folder):
    #         os.makedirs(arg.out_folder)
    #     for i in range(5):
    #         data_out_path = '{}/{}_{}_data_joint.npy'.format(arg.out_folder, i, p)
    #         label_out_path = '{}/{}_{}_label.pkl'.format(arg.out_folder, i, p)
    #         gendata('{}_{}'.format(i, p), arg.data_path, data_out_path, label_out_path)
