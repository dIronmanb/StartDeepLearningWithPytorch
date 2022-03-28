import os
import yaml
import torch.nn.functional as F
import torch
import numpy as np
import datetime
from pytz import timezone

from torch.utils.tensorboard import SummaryWriter
import torchvision

import src.optimizers.loss as loss
import src.optimizers.scheduler as scheduler
import src.optimizers.optimizer as optim
import src.dataloader.dataloader as dataloader
import src.models.model as model
import src.metrics.metrics as metric # 정확도 판단
import test


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 
current_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H:%M:%S')

# Open config file
with open("config/config.yaml", 'r', encoding = 'utf-8') as stream:
    try:
        config = yaml.safe_load(stream) # return into Dict
    except yaml.YAMLError as exc:
        print(exc)    

# Load data
train_data, valid_data, test_data = dataloader.get_data()

# Use cuda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# make model
vgg = model.get_VGG(config).to(DEVICE)
if(config['is_trained']):
    vgg.load_state_dict(torch.load('save/VGG11/2022-03-24_11:02:16/saved_weights_198'))

# print model
print(vgg)

# set epoch
epoch = config["epoch"]

# set loss and optimizer
criterion = loss.get_loss(config)
optimizer = optim.get_optimizer(vgg.parameters(), config)
schedule = scheduler.get_scheduler(optimizer, config)


# convert vgg to train_mode
vgg.train()

# show images in tensorboard (as batch size)
writer = SummaryWriter('runs/' + config['model'] + '_' + current_time)
dataiter = iter(train_data)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('Fruit_images', img_grid)


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
        outputs = vgg(data)
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
            
            for val_input, val_label in valid_data:
                
                val_input = val_input.to(DEVICE)
                val_label = val_label.to(DEVICE)
                val_outputs = vgg(val_input)
                val_loss = criterion(val_outputs, val_label)
                
                _ , val_preds = torch.max(val_outputs, 1)
                validation_running_loss += val_loss.item()
                validation_running_correct += torch.sum(val_preds == val_label.data)
                
                
    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_correct.float() / len(train_data)
    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_acc)
    
    val_epoch_loss = validation_running_loss / len(valid_data)
    val_epoch_acc = validation_running_correct.float() / len(valid_data)
    validation_running_loss_history.append(val_epoch_loss)
    validation_running_correct_history.append(val_epoch_acc)
    
    
    print("===================================================")
    print("epoch: ", i + 1)
    print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
    print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))
    
    writer.add_scalar("Loss/train_epoch", epoch_loss, i)
    writer.add_scalar("Accuracy/train_epcoh", epoch_acc, i)
    writer.add_scalar("Loss/valid_epoch", val_epoch_loss , i)
    writer.add_scalar("Accuracy/valid_epoch",val_epoch_acc, i) 
    
    if i % 10 == 0:
        if os.path.exists('save/' + config['model'] + '/' + current_time):
            pass   
        else:
            os.makedirs('save/' + config['model'] + '/' + current_time)
        
        torch.save(vgg.state_dict(), 'save/' + config['model'] + '/' + current_time + '/saved_weights_' + str(i))

        
            
# test and metric
# accuracy = test.predict(vgg, test_data, config)
# print('The test accuracy: {0:.3f}%'.format(accuracy))


