'''
    train.py
    모든 .py파일을 import하는 곳이다.
    데이터를 로드하는 것부터 모델 생성, 손실함수 및 옵티마이저까지.
    먼저 각 data_loader.py, model.py, loss.py, optimizer.py를 독립적으로 구현하고 여기로 다시 오자.
    
    - model.py에서만 class불러오고 나머지는 함수만 사용

'''

import os
import yaml
import torch.nn.functional as F
import torch
import numpy as np
import datetime
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

import optimizers.loss as loss_function
import optimizers.optimizer as optim
import optimizers.scheduler as scheduler
import data_loaders.data_loaders as data_loader
import model.models as models
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
train_data, test_data, classes = data_loader.data_load(config)


# Use cuda
DEVICE = torch.device("cuda" if config['use_cuda'] else "cpu")

# make model
model = models.get_cnn_model(config).to(DEVICE)
# model = model.to(DEVICE)
print(model)


# set epoch
epoch= config["epoch"]

# set loss and optimizer
criterion = loss_function.get_loss_function(config['loss_function'])
optimizer = optim.get_optimizer(model.parameters(), config)
schedule = scheduler.get_scheduler(optimizer, config)

# convert model to train_mode
model.train()

writer = SummaryWriter('runs/' + config['file_name'] + '_' + config['model'] + '_' + current_time)
dataiter = iter(train_data)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('32_CIFAR10_images', img_grid)

# train model           
running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []

for i in range(1, epoch + 1):
    running_loss = 0.0
    running_correct = 0.0
    validation_running_loss = 0.0
    validation_running_correct = 0.0
    
    total_loss = 0.0   # 배치에서의 loss
    total_length = 0.0 # 현재 길이
    for batch_idx, (data, targets) in enumerate(train_data):
    
        data = data.to(DEVICE)
        targets = targets.to(DEVICE, dtype = torch.int64)
        outputs = model(data)
        los = criterion(outputs, targets)
        
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        
        _ , preds = torch.max(outputs, 1)
        
        running_correct += torch.sum(preds == targets.data)
        running_loss += los.item()
        
        total_loss += los.item() * len(data)
        total_length += len(data)
        
        if batch_idx % 100 == 0:
            writer.add_scalar("Loss/train_step",los.item(), batch_idx + len(data) * (i))
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    i + 1, batch_idx * len(data), len(train_data.dataset), 100. * batch_idx / len(train_data), los.item()        
        ))
        
    else:
        
        with torch.no_grad():
            
            for val_input, val_label in test_data:
                
                val_input = val_input.to(DEVICE)
                val_label = val_label.to(DEVICE)
                val_outputs = model(val_input)
                val_loss = criterion(val_outputs, val_label)
                
                _ , val_preds = torch.max(val_outputs, 1)
                validation_running_loss += val_loss.item()
                validation_running_correct += torch.sum(val_preds == val_label.data)
                
                
    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_correct.float() / len(train_data)
    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_acc)
    
    val_epoch_loss = validation_running_loss / len(test_data)
    val_epoch_acc = validation_running_correct.float() / len(test_data)
    validation_running_loss_history.append(val_epoch_loss)
    validation_running_correct_history.append(val_epoch_acc)
    
    
    print("===================================================")
    print("epoch: ", i + 1)
    print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
    print("test loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))
    
    writer.add_scalar("Loss/train_epoch", epoch_loss, i)
    writer.add_scalar("Accuracy/train_epcoh", epoch_acc, i)
    writer.add_scalar("Loss/test_epoch", val_epoch_loss , i)
    writer.add_scalar("Accuracy/test_epoch",val_epoch_acc, i) 
    
    if i % 10 == 0:
        if os.path.exists('save/' + config['model'] + '/' + current_time):
            pass   
        else:
            os.makedirs('save/' + config['model'] + '/' + current_time)
        
        torch.save(model.state_dict(), 'save/' + config['model'] + '/' + current_time + '/saved_weights_' + str(i))

              
# test and metric
accuracy = test.predict(model, test_data, config)
print('The test accuracy: {0:.3f}%'.format(accuracy))
# met = metric.get_metrics(test_data[1], np.squeeze(predicted_list), config)
# print('The test accuracy: {}%'.format(met * 100))
