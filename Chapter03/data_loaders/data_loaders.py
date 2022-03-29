'''
 이미지 데이터는 촬영된 환경에 따라 명도나 채도 등이 서로 모두 다르기 때문에
 영상 기반 딥러닝 모델을 학습시키기 전에 모든 이미지들을 동일한 환경으로 맞춰주는 게 중요!
 >> 전체 이미지에 대한 화소 값의 mean과 standard deviation을 구하여 일괄 적용해야 함.
 >> Imagenet 데이터 세트에서 계산된 평균과 표준 편차 사용


 torch.Tensor(): T = torch.Tensor() 문장 입력시 T는 tensor자료구조 클래스 생성
 torch.tensor(): 어떤 data를 tensor로 copy, 괄호 안에 값이 없다면 에러남
 
 train_set과 test_set모두에 preparation을 진행하는 이유:
    The training and testing data should undergo the same data preparation steps 
    or the predictive model will not make sense. 
    This means that the number of features for both the training and test set 
    should be the same and represent the same thing.
'''

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np       


def data_load(config):
    name = config['data_loader_name']
    if name == 'data_load_only_normalizing':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4913, 0.4821, 0.4465],
                                 std = [0.2470, 0.2434, 0.2615])
        ])
    elif name == 'data_load_normalizing_and_agumentation':
        
        data_transform = transforms.Compose([                                                                                    
            transforms.RandomHorizontalFlip(p=0.5),       # 이미지를 좌우반전
            # transforms.RandomVerticalFlip(p=0.5),       # 이미지를 상하반전
            transforms.RandomRotation(10),    # 이미지를 -90 ~ 90에서 랜덤하게 rotate
            # transforms.RandomResizedCrop(size = (32,32))      # 이미지의 임의의 부분을 확대하여 Resize
            # transforms.ToPILImage(mode = None):     # PILImage로 변환

            transforms.ToTensor(),               # 이미지를 tensor_type로 변환
            # transforms.RandomApply(transforms = data_transform, p = 1.0),                 # agumentation을 전체 데이터에서 임의적으로 골라서 취함.                
            # 왜 0.5, 0.5, 0.5를 넣어야 하는가?
            # >> 직접 mean과 std를 구하면 mean = [0.49139968 0.48215841 0.44653091], std = [0.24703223 0.24348513 0.26158784]
            # channel은 3개
            transforms.Normalize(mean = [0.4913, 0.4821, 0.4465], std = [0.2470, 0.2434, 0.2615]) # tensor의 데이터 수치를 정규화한다.
                                                                                # transforms.Normalize((0.5), (0.5))) -> -1 ~ 1 사이의 값으로 normalized                                                                  # output[channel] = (input[channel] - mean[channel]) / std[channel]  
            ])
    elif name == 'data_load_rainbow':
            data_transform = transforms.Compose([                                                                                    
            transforms.RandomHorizontalFlip(p=0.5),       # 이미지를 좌우반전
            transforms.RandomRotation(10),    # 이미지를 -90 ~ 90에서 랜덤하게 rotate

            transforms.ToTensor(),               # 이미지를 tensor_type로 변환
            # transforms.RandomApply(transforms = data_transform, p = 1.0),                 # agumentation을 전체 데이터에서 임의적으로 골라서 취함.                
            # 왜 0.5, 0.5, 0.5를 넣어야 하는가?
            # >> 직접 mean과 std를 구하면 mean = [0.49139968 0.48215841 0.44653091], std = [0.24703223 0.24348513 0.26158784]
            # channel은 3개
            transforms.Normalize(mean = [0.4913, 0.4821, 0.4465], std = [0.2470, 0.2434, 0.2615]) # tensor의 데이터 수치를 정규화한다.
                                                                                # transforms.Normalize((0.5), (0.5))) -> -1 ~ 1 사이의 값으로 normalized                                                                  # output[channel] = (input[channel] - mean[channel]) / std[channel]  
            ])
        
    else:
        print("There was no name in DataLoader_Name")
        
        
    train_set = datasets.CIFAR10(root = '/data/Github_Management/StartDeepLearningWithPytorch/Chapter03/cifar10',
                                 train = True,
                                 download = True,           # If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
                                 transform = data_transform)
    
    test_set = datasets.CIFAR10(root = '/data/Github_Management/StartDeepLearningWithPytorch/Chapter03/cifar10',
                                train = False,
                                download = True,
                                transform = data_transform
                                )
      
    # print(train_set.data.mean(axis = (0,1,2)) / 255)
    # print(train_set.data.std(axis = (0,1,2))  / 255)
    
    # print(test_set.data.mean(axis = (0,1,2)) / 255)
    # print(test_set.data.std(axis = (0,1,2))  / 255)
        
    train_loader = DataLoader(train_set,
                              batch_size= 64, #['train_batch_size'],
                              num_workers = config['num_workers'],
                              shuffle = True)#config['train_dataset_shuffle'])
    
    test_loader = DataLoader(test_set,
                             batch_size = 64, #config['test_batch_size'],
                             num_workers = config['num_workers'],
                             shuffle = False) #config['test_dataset_shuffle']) 
    
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes # train
    
    
    