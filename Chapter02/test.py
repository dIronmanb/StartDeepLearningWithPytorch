import torch

def predict(model, test_images_tensor, config):
    model.eval()
    result = []
    DEVICE = torch.device("cuda" if config['use_cuda'] else "cpu")
    
    with torch.no_grad(): # 그래디언트를 구하지 않겠다 = 가중치값을 변화하지 않겠다
        for data in test_images_tensor:
            data = data.to(DEVICE)
            output = model(data.view(-1, 3, 32, 32))
            prediction = output.max(1, keepdim = True)[1]
            result.append(prediction.tolist())
    return result