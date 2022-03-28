import torch
import yaml
import data_loaders.data_loaders as dataloader
import model.models as vgg

def predict(model, test_loader, config):
    model.eval()
    
    total = 0
    correct = 0
    
    DEVICE = torch.device("cuda" if config['use_cuda'] else "cpu")
    
    
    with torch.no_grad(): # 그래디언트를 구하지 않겠다 = 가중치값을 변화하지 않겠다
        for (images, labels) in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _ , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == '__main__':
    
    # get config file
    print("Get config file...")
    with open("/data/Github_Management/StartDeepLearningWithPytorch/Chapter03/config.yaml", 'r', encoding = 'utf-8') as stream:
        try:
            config = yaml.safe_load(stream) # return into Dict
        except yaml.YAMLError as exc:
            print(exc) 
            
    # load data
    print("Loading data...")
    train_data, test_data, _ = dataloader.data_load(config)
        
    # Use cuda
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create learned model
    print("Creating learned model...")
    model = vgg.get_cnn_model(config).to(DEVICE)
    
    # load weights >> 디렉토리에서 가중치 파일 불러오기
    model.load_state_dict(torch.load('save/VGG11/2022-03-24_23:20:09/saved_weights_100'))
    
    acc = predict(model, test_data, config)
    print("The accuracy is {:.1f}%".format(acc))
