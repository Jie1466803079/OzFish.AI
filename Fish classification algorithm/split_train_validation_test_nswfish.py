import os
import numpy as np
import shutil

path_train = "E:\\Jie\\998\\Model_v2\\OneDrive_2022-07-31\\NSWFish_train"
path_validation = "E:\\Jie\\998\\Model_v2\\OneDrive_2022-07-31\\NSWFish_validation"
path_test = "E:\\Jie\\998\\Model_v2\\OneDrive_2022-07-31\\NSWFish_test"

ori_file = "E:\\Jie\\998\\Model_v2\\OneDrive_2022-07-31\\NSW_Fish_Dataset"

ratio01 = 0.6
ratio02 = 0.8

names = os.listdir(ori_file)

for name in names:
    if name not in os.listdir(path_train):
        os.mkdir(os.path.join(path_train, name))
    ori_path = os.listdir(os.path.join(ori_file, name))
    num_train = int(len(ori_path) * ratio01)
    num_train_and_validation = int(len(ori_path) * ratio02)
    np.random.shuffle(ori_path)
    train_image = ori_path[:num_train]
    validation_image = ori_path[num_train:num_train_and_validation]
    test_image = ori_path[num_train_and_validation:]

    for img in train_image:
        source = os.path.join(ori_file, name, img)
        target = os.path.join(path_train, name, img)
        shutil.copy(source, target)

    if name not in os.listdir(path_validation):
        os.mkdir(os.path.join(path_validation, name))
    for img2 in validation_image:
        source2 = os.path.join(ori_file, name, img2)
        target2 = os.path.join(path_validation, name, img2)
        shutil.copy(source2, target2)


    if name not in os.listdir(path_test):
        os.mkdir(os.path.join(path_test, name))
    for img1 in test_image:
        source1 = os.path.join(ori_file, name, img1)
        target1 = os.path.join(path_test, name, img1)
        shutil.copy(source1, target1)

