#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import pickle

#np.set_printoptions(threshold=np.inf) #打印时去掉省略号

def takeSecond(elem):
    # 功能：获取列表的第二个元素
    return elem[1]

def takeOne(elem):
    # 功能：获取列表的第一个元素
    return elem[0]

#data_path = 'F:\EEG_dataset\DEAP_set\data_preprocessed_python'
def read_data(data_path, remove_baseline):
    """
    # 功能：读取32人EEG数据并去除基线信号
    :param:
        data_time: 60s刺激时间, 截取中间 data_time 时长
        remove_baseline: True移除基线, False不移除基线
    :return:
        data: 4D tensor of [32,40,32,7680]-[people,video,channel,data]
        labels: 3D tensor of [32,40,4]-[people,video,label]
        label_index: [0,1,2,3]-[valence,arousal,dominance,liking]
    """
    data = []
    labels = []
    filenames = [os.path.join(data_path, 's%.2d.dat' % i) for i in range(1, 33)]
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)

    for filename in filenames:
        with open(filename, 'rb') as f:
            # print(filename)
            array = pickle.load(f, encoding='latin1') # array = {b'data': ~, b'labels': ~}
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
            # 然后返回由这些元组组成的对象
            for datum, label in zip(list(array["data"]), list(array["labels"])):
                data.append(np.array(datum).flatten())
                labels.append(np.array(label).flatten())
    data = np.reshape(data, [32,40,40,8064])
    data.astype(np.float32) # 将 float64 转换为 float32

    # 选取32通道EEG信号
    data = data[:,:,:32,:] # shape=(32,40,32,8064)
    labels1 = np.reshape(labels, [32,40,4])
    labels = np.int64(labels1 > 5) # 阈值设为5，>5为高水平1，<=5为低水平0

    if remove_baseline==0: # 不去除基线
        return data[:,:,:,384:], labels  # shape=[32,40,32,7680], [32,40,4]

    elif remove_baseline==1: # 去除基线，均值法
        # 求出基线信号均值，3s基线信号，采样频率128Hz
        base_signal = (data[:, :, :, 0:128] + data[:, :, :, 128:256] + data[:, :, :, 256:384]) / 3
        for i in range(3, 63):
            data[:,:,:,i*128:(i+1)*128] = data[:,:,:,i*128:(i+1)*128] - base_signal
        return data[:,:,:,384:], labels # shape=[32,40,32,7680], [32,40,4]

    elif remove_baseline==2: # 去除基线，均值升级
        """
        base_signal = (data[:, :, :, 64:192] + data[:, :, :, 192:320]) / 2
        for i in range(3, 63):
            data[:,:,:,i*128:(i+1)*128] = data[:,:,:,i*128:(i+1)*128] - base_signal
        return data[:,:,:,384:], labels # shape=[32,40,32,7680], [32,40,4]
        """
        base_signal = np.zeros([32, 40, 32, 128], dtype=np.float32)
        for j in range(128):
            base_signal[:,:,:,j] = np.sum(data[:,:,:,j*3:(j+1)*3], axis=3) / 3
        for i in range(3, 63):
            data[:,:,:,i*128:(i+1)*128] = data[:,:,:,i*128:(i+1)*128] - base_signal
        return data[:,:,:,384:], labels # shape=[32,40,32,7680], [32,40,4]

    elif remove_baseline == 4: # 去除基线，截取2秒
        base_signal_all = data[:,:,:,0:384]
        data_video = data[:,:,:,384:]
        base_signal = np.zeros([32,40,32,128], dtype=np.float32)
        for people in range(32):
            for video in range(40):
                for channel in range(32):
                    tmp = list(enumerate(base_signal_all[people][video][channel]))
                    tmp.sort(key=takeSecond)  # 指定第二个元素的从小到大排序
                    tmp_slip = tmp[64:320] # 截取中间256个数值
                    tmp_slip_tmp = [tmp_slip[i] for i in range(0,256,2)]
                    tmp_slip_tmp.sort(key=takeOne)  # 指定第一个元素的从小到大排序
                    base_signal[people][video][channel] = [value for index, value in tmp_slip_tmp]
        for i in range(60):
            data_video[:,:,:,i*128:(i+1)*128] = data_video[:,:,:,i*128:(i+1)*128] - base_signal
        return data_video, labels  # shape=[32,40,32,7680], [32,40,4]


def data_labels(data_path, remove_baseline):
    # 功能：将所有数据帧和标签添加到一个列表中
    # shape=[32,40,32,7680], [32,40,4]
    data, labels = read_data(data_path, remove_baseline)
    data_label_list = []
    for people in range(32):
        for video in range(40):
            for segment in range(60):
                frame = data[people][video][:,segment*128:segment*128+128] #(32,128)
                label = labels[people][video] # (4,)
                #label_tmp = np.int64(label > 5)  # (4,)阈值设为5，>5为高水平1，<=5为低水平0

                frame_list = frame.flatten().tolist()
                frame_list.extend(label) #后四位是标签

                data_label_list.append(frame_list)

    return data_label_list # len = 32*40*60 = 76800


def slip_list(list, start, end):
    # 功能：指定位置截取列表[start,end)
    list_tmp = []
    for i in range(start, end):
        list_tmp.append(list[i])
    return list_tmp

def slip_del_list_end(list, k):
    # 功能：分离列表后k个元素
    end_list = []
    for i in range(k):
        end_list.insert(0, list[-1])
        list.pop()
    return list, end_list


def train_test_set(data_path, remove_baseline):
    # 功能：随机打乱生成一个总的数据集
    # len=32x40x60=76800--32x128+4
    data_label_list = data_labels(data_path, remove_baseline)
    total_data = []
    total_labels = []
    for people in range(32):
        start = people * 40 * 60
        end = (people + 1) * 40 * 60
        people_per = slip_list(data_label_list, start, end) #每个人的全部数据，(40x60--32x128+4)

        # random.sample(list, k)从list中随机获取k个元素，作为一个片断返回
        random_people_per = random.sample(people_per, 40*60) # 打乱顺序，(40x60--32x128+4)

        for frame_list in random_people_per:
            # 分离每一帧的数据和标签，frame_train=(32x128), label_train=(4)
            frame_train, label_train = slip_del_list_end(frame_list, 4)
            total_data.append(frame_train)
            total_labels.append(label_train)

    total_data_array = np.array(total_data).reshape([32, 40 * 60, 32, 128])
    total_labels_array = np.array(total_labels).reshape([32, 40 * 60, 4])
    return total_data_array, total_labels_array

"""
# GPU加速
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 或者export CUDA_VISIBLE_DEVICES=0
data_path = 'E:\python\CNN_deap\data_preprocessed_python' # 加载地址
filename = "E:\python\CNN_deap\CNN_deap_set\with_random" # 存储地址
for i in range(2,3):
    data, labels= train_test_set(data_path=data_path, remove_baseline=i)
    np.save(filename + "\\data_%d.npy" % i, data) # shape=[32,2400,32,128]
    np.save(filename + "\\labels_%d.npy" % i, labels) # shape=[32,2400,4]
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 或者export CUDA_VISIBLE_DEVICES=0
data_path = 'E:\python\CNN_deap\data_preprocessed_python' # 加载地址
filename = "E:\python\CNN_deap\CNN_deap_set\without_random" # 存储地址
for i in range(4,5):
    data, labels= read_data(data_path=data_path, remove_baseline=i)
    np.save(filename + "\\data_%d.npy" % i, data) # shape=[32,40,32,7680]
    if i ==0: np.save(filename + "\\labels.npy", labels) # shape=[32,40,4]
    else: continue
""""""