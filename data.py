import torch
from torchvision import datasets, transforms
import numpy as np

class ImageSequence(datasets.MNIST):
    def __init__(self, root, seq_len, batch_size, delay, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.delay = delay
        self.labels = load_label(train)

    def __getitem__(self, index):
        x = np.array(self.data[index])
        batch_index = index // self.batch_size
        if index % self.batch_size == 0:
            index = batch_index * (self.batch_size - self.seq_len)
        else:
            index = batch_index * (self.batch_size - self.seq_len) + (index % self.batch_size) - self.seq_len
        if index > len(self.labels) or index == len(self.labels):
            index = 0
        index = index - self.delay
        return self.transform(x), self.labels[index]
    
    def __len__(self):
        return len(self.data)


def get_data(slidingWindow, batch_size, delay):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    data_train = ImageSequence(root = "../data/",
                                seq_len=slidingWindow,
                                batch_size = batch_size,
                                delay=delay,
                                transform=transform,
                                train = True,
                                download = True)

    data_test = ImageSequence(root="../data/",
                            seq_len=slidingWindow,
                            batch_size = batch_size,
                            delay=delay,
                            transform = transform,
                            train = False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size = batch_size,
                                                    shuffle = False)

    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size = batch_size,
                                                shuffle = False)

    return train_loader, test_loader


def get_raw_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    data_train = datasets.MNIST(root = "../data/",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.MNIST(root="../data/",
                            transform = transform,
                            train = False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size = batch_size,
                                                    shuffle = False)

    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size = batch_size,
                                                shuffle = False)

    return train_loader, test_loader

def create_matrix(slidingWindow):  
    mat = np.random.rand(slidingWindow)
    print(mat)
    time_matrix = torch.from_numpy(mat.reshape(1, slidingWindow))
    return time_matrix

def gen_result(time_matrix, labels, slidingWindow):
    res = torch.mm(time_matrix, labels.resize(slidingWindow,1).double())    
    return res

def gen_label(slidingWindow, batch_size, reset=False):
    if reset:
        matrix = create_matrix(slidingWindow)

        train_loader, test_loader = get_raw_data(batch_size)
        perior_result = torch.zeros([0,1])
        for batch_index, (_, labels) in enumerate(train_loader):
            for i in range(len(labels)-slidingWindow):
                result = gen_result(matrix, labels[i:i+slidingWindow], slidingWindow)
                result = result.float()
                perior_result = torch.cat((perior_result,result.view(1,1)))
        labels = perior_result.detach().numpy()
        max = labels.max()
        min = labels.min()
        for i in range(len(labels)):
            labels[i] = (labels[i]-min)/(max-min)
        print("mean",np.mean(labels))
        print("var",np.var(labels))
        np.savetxt('./train_labels.csv', labels, fmt='%f')

        perior_result = torch.zeros([0,1])
        for batch_index, (_, labels) in enumerate(test_loader):
            for i in range(len(labels)-slidingWindow):
                result = gen_result(matrix, labels[i:i+slidingWindow], slidingWindow)
                result = result.float()
                perior_result = torch.cat((perior_result,result.view(1,1)))
        labels = perior_result.detach().numpy()
        max = labels.max()
        min = labels.min()
        for i in range(len(labels)):
            labels[i] = (labels[i]-min)/(max-min)
        np.savetxt('./test_labels.csv', labels, fmt='%f')

def load_label(train):
    if train:
        labels = np.loadtxt('./train_labels.csv')
    else:
        labels = np.loadtxt('./test_labels.csv')
    return labels



