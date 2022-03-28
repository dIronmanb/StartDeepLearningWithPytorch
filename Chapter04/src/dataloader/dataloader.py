import os
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
# 전체 과일 종류 개수 출력하기
# file_list = os.listdir(path + 'train')
# print('total number of type of fruits:',len(file_list))


def get_data():
    train_dir = '/data/Github_Management/StartDeepLearningWithPytorch/Chapter04/dataset/train/'
    val_dir = '/data/Github_Management/StartDeepLearningWithPytorch/Chapter04/dataset/validation'
    test_dir = '/data/Github_Management/StartDeepLearningWithPytorch/Chapter04/dataset/test'

    # 과일 이름을 담은 리스트
    classes = os.listdir(train_dir)
    # print(classes)

    train_transform = transforms.Compose([
                            transforms.RandomRotation(10), # +/- 10 degrees
                            transforms.RandomHorizontalFlip(), # reverse 50% of images -> 위아로 filp은 X
                            transforms.Resize(40), # (40, 40)
                            transforms.CenterCrop(40), #(40, 40)
                            transforms.ToTensor(), # 텐서로 변환
                            transforms.Normalize(mean = [0.5, 0.5, 0.5], \
                                                std  = [0.5, 0.5, 0.5]) # mu와 std는 나중에 구해보기                                             
                        ])

    train_set = ImageFolder(train_dir, transform = train_transform)
    valid_set = ImageFolder(val_dir,   transform = train_transform)
    test_set = ImageFolder(test_dir, transform = train_transform)
    
    # Train, Valid, Test
    # num_data = [len(train_set), len(valid_set), len(test_set)]
    # print(num_data)
    print(type(train_set))
    print(type(valid_set))

    train_loader = DataLoader(train_set, batch_size = 64, num_workers = 2, shuffle = True)
    valid_loader = DataLoader(valid_set, batch_size = 64, num_workers = 2, shuffle = True)
    test_loader  = DataLoader(test_set, batch_size = 64, num_workers = 2, shuffle = False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # 전체 과일 종류 개수 출력하기
    file_list = os.listdir('/data/Github_Management/StartDeepLearningWithPytorch/Chapter04/dataset/train/')
    print('total number of type of fruits:',len(file_list))
    
    a = get_data()
