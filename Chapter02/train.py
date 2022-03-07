'''
    train.py
    모든 .py파일을 import하는 곳이다.
    데이터를 로드하는 것부터 모델 생성, 손실함수 및 옵티마이저까지.
    먼저 각 data_loader.py, model.py, loss.py, optimizer.py를 독립적으로 구현하고 여기로 다시 오자.
    
    - model.py에서만 class불러오고 나머지는 함수만 사용

'''


import yaml
import torch.nn.functional as F
import torch
import numpy as np
import datetime
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

import optimizers.loss as loss_function

import optimizers.optimizer as optim
import data_loaders
import models
import torchvision
import test
import metric

# 
current_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H:%M:%S')

# Open config file
with open("config.yaml", 'r', encoding = 'utf-8') as stream:
    try:
        config = yaml.safe_load(stream) # return into Dict
    except yaml.YAMLError as exc:
        print(exc)    

# Load data
train_data, test_data = data_loaders.data_load(config)

# Use cuda
DEVICE = torch.device("cuda" if config['use_cuda'] else "cpu")

# make model
model = models.get_cnn_model(config["model"]).to(DEVICE)
# model = model.to(DEVICE)
print(model)


# set epoch
epoch= config["epoch"]

# set loss and optimizer
criterion = loss_function.get_loss_function(config['loss_function'])
optimizer = optim.get_optimizer(model.parameters(), config)

# convert model to train_mode
model.train()

writer = SummaryWriter('runs/CIFAR10_experient_' + str(config['model']) + '_' + current_time)
dataiter = iter(train_data)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('32_CIFAR10_images', img_grid)

# train model
for i in range(epoch):
    total_loss = 0.0
    total_length = 0
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(DEVICE), target.to(DEVICE, dtype = torch.int64)
        
        predicted = model(data)
        loss = criterion(predicted, target)
        optimizer.zero_grad() # 경사하강법 하기 직전에 초기화 -> 안하면 이전의 그래디언트가 중첩된다.
        loss.backward() # 구한 loss로부터 back propagation을 통해 각 변수의 loss에 대한 편미분을 한다.
        optimizer.step() # 미리 선언할 때 지정한 model의 파라미터들이 업데이트된다.
        
        total_loss += loss.item() * len(data)
        total_length += len(data)
            
        if batch_idx % 100 == 0:
            writer.add_scalar("Loss/train_step",loss.item(), batch_idx + len(data) * (i))
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    i, batch_idx * len(data), len(train_data.dataset), 100. * batch_idx / len(train_data), loss.item()        
        ))
    writer.add_scalar("Loss/train_epoch", total_loss/total_length, i)
            
            
# test and metric
predicted_list = test.predict(model, test_data[0], config)
met = metric.get_metrics(test_data[1], np.squeeze(predicted_list), config)
print('The test accuracy: {}%'.format(met * 100))
