import torch
import yaml
import src.models.model as vgg
import src.dataloader.dataloader as dataloader
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
def predict(model, test_loader):
    model.eval()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get classes
    train_dir = '/data/Github_Management/StartDeepLearningWithPytorch/Chapter04/dataset/train/'
    classes = os.listdir(train_dir)
    
    # get tensorboard
    writer = SummaryWriter('final_result/test_01')    
    
    
    with torch.no_grad(): # 그래디언트를 구하지 않겠다 = 가중치값을 변화하지 않겠다
        dataiter = iter(test_loader)
        images, labels = dataiter.next()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        _ , index = torch.max(outputs.data, 1)
        img_grid = torchvision.utils.make_grid(images)        
        
        for i in range(len(index.tolist())):      
            result = classes[index[i]]
            writer.add_image('The predicted result is ' + result, img_grid)


if __name__ == '__main__':
    
    # get config file
    print("Get config file...")
    with open("config/config.yaml", 'r', encoding = 'utf-8') as stream:
        try:
            config = yaml.safe_load(stream) # return into Dict
        except yaml.YAMLError as exc:
            print(exc) 
            
            
    # load data
    print("Loading data...")
    train_data, valid_data, test_data = dataloader.get_data()
        
    # Use cuda
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create learned model
    print("Creating learned model...")
    model = vgg.get_VGG(config).to(DEVICE)
    
    # load weights >> 디렉토리에서 가중치 파일 불러오기
    model.load_state_dict(torch.load('save/VGG11/2022-03-24_13:32:10/saved_weights_297'))
    
    predict(model, test_data)
    
        
    
    
    