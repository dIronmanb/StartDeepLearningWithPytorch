import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

def pickle_to_images_and_labels(root):
    # Get dataset
    data = unpickle(root)
    # normalize
    data_images = data[b'data'] / 255 
    # reshape data >> -1 * 3 * 32 * 32
    # num of data, colour, width, height
    data_images = data_images.reshape(-1, 3, 32, 32).astype("float32")
    data_labels = data[b'labels']
    return data_images, data_labels

def data_load(config):
    # preprocessing data
    train_images_and_labels = []
    for i in range(5):
        train_images_and_labels.append(pickle_to_images_and_labels("/data/CNN_model/dataset/cifar-10-batches-py/data_batch_" + str(i + 1)))

    train_images = np.concatenate([i[0] for i in train_images_and_labels], axis = 0)
    train_labels = np.concatenate([i[1] for i in train_images_and_labels], axis = 0)

    test_images, test_labels = pickle_to_images_and_labels("/data/CNN_model/dataset/cifar-10-batches-py/test_batch")
    test_images = np.concatenate([test_images], axis = 0)
    test_labels = np.concatenate([test_labels], axis = 0)


    # Convert numpy to tensor
    train_images_tensor = torch.tensor(train_images)
    train_labels_tensor = torch.tensor(train_labels)
    train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
    train_loader = DataLoader(train_tensor, 
                              batch_size = config['batch_size'], 
                              num_workers=config['num_workers'], 
                              shuffle = config['dataset_shuffle']
                             )

    test_images_tensor = torch.tensor(test_images)
    
    test_loader = (test_images_tensor, test_labels)
    return train_loader, test_loader
    
    
    