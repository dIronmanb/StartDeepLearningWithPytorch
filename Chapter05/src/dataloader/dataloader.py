from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np       

# datasets.ImageFolder

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
            transforms.RandomRotation(10),
            # Image (가로, 세로, 채널)
            transforms.ToTensor(), # (채널, 세로, 가로)
            transforms.Normalize(mean = [0.4913, 0.4821, 0.4465], std = [0.2470, 0.2434, 0.2615]) # tensor의 데이터 수치를 정규화한다.
            ])
        
    elif name == 'data_load_rainbow':
            data_transform = transforms.Compose([                                                                                    
            transforms.RandomHorizontalFlip(p=0.5),       # 이미지를 좌우반전
            transforms.RandomRotation(10),    
            transforms.ToTensor(),  
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
    
    
    