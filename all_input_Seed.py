# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import random

np.set_printoptions(threshold=np.inf) #打印时去掉省略号

def z_score_norm(data):
    # Z-Score标准化，data二维矩阵
    mean = data.mean()
    sigma = data.std()
    data = (data - mean)/sigma
    return data

def random_swap(train_batch, label_image_list, channel_list, seedT):
    swap_count = len(label_image_list) // 4  # 标签0交换的次数
    for num in range(swap_count):
        choose_2images = random.sample(label_image_list, 2)
        image_1 = choose_2images[0]  # shape=(1,)
        image_2 = choose_2images[1]
        #swap_channel_num = random.randint(14, 23)  # 交换多少个通道
        swap_channel_num = seedT  # 交换多少个通道
        swap_channel_list = random.sample(channel_list, swap_channel_num)
        for channel in swap_channel_list:
            tmp = train_batch[image_1][channel]  # shape=(128,)
            train_batch[image_1][channel] = train_batch[image_2][channel]
            train_batch[image_2][channel] = tmp

def random_channel_swap(train_batch, train_labels, seedT):
    # 功能：随机做通道交换, 针对的是时序图
    # train_batch=[batch_size, channels, samples], train_labels=[batch_size]
    label_is0_image_list = [index for index, value in enumerate(train_labels) if value == 0]
    label_is1_image_list = [index for index, value in enumerate(train_labels) if value == 1]
    channel_list = [i for i in range(train_batch.shape[1])]

    random_swap(train_batch, label_is0_image_list, channel_list, seedT)
    random_swap(train_batch, label_is1_image_list, channel_list, seedT)
    return train_batch # shape=[batch_size, channels, samples]

# 生成训练批次
def return_train_set(file_path, batch_size, window_size, label_index, method, fold, seedT):
    # label_index: 标签的下标，选择分类类别，与测试集保持严格相同
    # label_index: [0,1,2,3]-[valence,arousal,dominance,liking]
    total_data = np.load(file_path + "\\data_%d.npy" % method) # shape=[32, 40, 32, 7680]
    total_labels = np.load(file_path + "\\labels.npy") # shape=[32, 40, 4]
    #total_data = total_data[:2, :, :, :] # 暂用前2人测试
    # 10折交叉验证
    if fold == 1:   train_data = total_data[:, :, :, :128 * 42]
    elif fold == 2: train_data = total_data[:, :, :, 128 * 6:128 * 48]
    elif fold == 3: train_data = total_data[:, :, :, 128 * 12:128 * 54]
    elif fold == 4: train_data = total_data[:, :, :, 128 * 18:]
    elif fold == 5: train_data = np.delete(total_data, [i for i in range(128 * 6, 128 * 24)], axis=3)
    elif fold == 6: train_data = np.delete(total_data, [i for i in range(128 * 12, 128 * 30)], axis=3)
    elif fold == 7: train_data = np.delete(total_data, [i for i in range(128 * 18, 128 * 36)], axis=3)
    elif fold == 8: train_data = np.delete(total_data, [i for i in range(128 * 24, 128 * 42)], axis=3)
    elif fold == 9: train_data = np.delete(total_data, [i for i in range(128 * 30, 128 * 48)], axis=3)
    elif fold == 10: train_data = np.delete(total_data, [i for i in range(128 * 36, 128 * 54)], axis=3)
    frame_group = []
    labels_group = []
    while(True):
        for i in range(batch_size):
            people = random.randint(0, 31)
            video = random.randint(0, 39)
            i = random.randint(0, 41)
            segment = train_data[people, video, :, i*window_size:i*window_size+window_size] #(32, window_size)
            train_batch_tmp = z_score_norm(segment) # shape=(32, window_size)

            frame_group.append(train_batch_tmp.flatten())
            labels_group.append(total_labels[people, video, label_index])

        train_batch = np.array(frame_group).reshape([batch_size, 32, window_size])
        label_batch = np.array(labels_group).reshape([batch_size])
        train_batch = random_channel_swap(train_batch, label_batch, seedT) # 通道交换
        frame_group.clear()
        labels_group.clear()
        # yield的效果和return相同，但是yield会记住此时的位置等待下一次调用，方式如下：
        # handle = return_train_set(data_path)
        # train_data, train_labels = next(handle)
        yield train_batch, label_batch # shape=[batch_size, 32, window_size]

# 生成测试批次
def return_test_set(file_path, batch_size, window_size, label_index, method, fold):
    # label_index: 标签的下标，选择分类类别，与测试集保持严格相同
    # label_index: [0,1,2,3]-[valence,arousal,dominance,liking]
    total_data = np.load(file_path + "\\data_%d.npy" % method) # shape=[32, 40, 32, 7680]
    total_labels = np.load(file_path + "\\labels.npy") # shape=[32, 40, 4]
    #total_data = total_data[:2, :, :, :] # 暂用前2人测试
    # 10折交叉验证
    if fold == 1:   test_data = total_data[:, :, :, 128 * 42:]
    elif fold == 2: test_data = np.delete(total_data, [i for i in range(128 * 6, 128 * 48)], axis=3)
    elif fold == 3: test_data = np.delete(total_data, [i for i in range(128 * 12, 128 * 54)], axis=3)
    elif fold == 4: test_data = total_data[:, :, :, :128 * 18]
    elif fold == 5: test_data = total_data[:, :, :, 128 * 6:128 * 24]
    elif fold == 6: test_data = total_data[:, :, :, 128 * 12:128 * 30]
    elif fold == 7: test_data = total_data[:, :, :, 128 * 18:128 * 36]
    elif fold == 8: test_data = total_data[:, :, :, 128 * 24:128 * 42]
    elif fold == 9: test_data = total_data[:, :, :, 128 * 30:128 * 48]
    elif fold == 10: test_data = total_data[:, :, :, 128 * 36:128 * 54]
    frame_group = []
    labels_group = []
    while(True):
        for people in range(32):
            for video in range(40):
                for i in range(18):
                    segment = test_data[people, video, :, i*window_size:(i+1)*window_size] #(32, window_size)
                    test_batch_tmp = z_score_norm(segment)  # shape=(32, window_size)
                    frame_group.append(test_batch_tmp.flatten())
                    labels_group.append(total_labels[people, video, label_index])

                    if len(frame_group) == batch_size:
                        test_batch = np.array(frame_group).reshape([batch_size, 32, window_size])
                        label_batch = np.array(labels_group).reshape([batch_size])
                        frame_group.clear()
                        labels_group.clear()
                        yield test_batch, label_batch # shape=[batch_size, 32, window_size]


"""
file_path = "E:\python\CNN_deap\CNN_deap_set\without_random"
a = return_test_set(file_path,32,128,0,4,1)
for i in range(10):
    data, labels = next(a)
    print("step = %d :" % (i+1))
    print(data.shape)
    print(labels)
"""
