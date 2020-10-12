import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import sys

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
        # index = index + 1 - self.seq_len
        # index = index - self.delay
        # index = index % len(self.labels)
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

# def createTestData(labelarray):
#     transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Lambda(lambda x: x.repeat(3,1,1)),
#                                 transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

#     data_test = datasets.MNIST(root="../data/",
#                             transform = transform,
#                             train = False)

#     test_loader = torch.utils.data.DataLoader(dataset=data_test,
#                                                 batch_size = 1000,
#                                                 shuffle = False)
    
#     j = 0   #当前查找到labelarray的第几个元素
#     imageArray = np.zeros((0,3,28,28))
#     for  (images, labels) in test_loader:
#         for i in range(len(images)):
#             if labels[i] == labelarray[j]:
#                 imageArray = np.vstack((imageArray,images[i].view(1,3,28,28).numpy()))
#                 if(j==len(labelarray)-1):
#                     # print(imageArray[-1].shape)
#                     # pil_img = Image.fromarray(imageArray[-2][1]*-255)
#                     # pil_img.show()
#                     print(imageArray.shape)
#                     return imageArray
#                 else:
#                     j = j+1
        
class Matrix:
    def __init__(self,n,m):     #n是状态空间维度，m是输出维度
        self.A = torch.randn(n,n)/n
        self.B = torch.randn(n,1)/n
        self.C = torch.randn(m,n)
        self.D = torch.randn(m,1)
        self.x = torch.randn(n,1)/10
        self.x0 = self.x

    def iter(self,u):
        u = u.float()/10
        # u-=5
        u = u.view(1,1)
        y = torch.mm(self.C,self.x) + torch.mm(self.D,u)         #y(t) = Cx(t) + Du(t)
        self.x = torch.mm(self.A,self.x) + torch.mm(self.B,u)    #x(t+1) = Ax(t) + Bu(t)
        return y

    def reset(self):
        self.x = self.x0

def create_matrix(slidingWindow):  
    mat = np.random.rand(slidingWindow)
    print(mat)
    time_matrix = torch.from_numpy(mat.reshape(1, slidingWindow))
    return time_matrix


def gen_label(n,m,batch_size,reset = False):
    if reset:
        matrix = Matrix(n,m)   
        train_loader, test_loader = get_raw_data(batch_size)
        result = torch.zeros([0,m])
        for batch_index, (_, labels) in enumerate(train_loader):
            for i in range(len(labels)):
                y = matrix.iter(labels[i])
                result = torch.cat((result,y))
        labels = result.detach().numpy()
        # max = labels.max()
        # min = labels.min()
        # for i in range(len(labels)):
            # labels[i] = (labels[i]-min)/(max-min)
        mean = np.mean(labels)
        std = np.std(labels)
        labels = (labels - mean) / std
        print("mean",mean)
        print("std",std)
        np.savetxt('./new_train_labels.csv', labels, fmt='%f')

        matrix.reset()
        result = torch.zeros([0,m])
        for batch_index, (_, labels) in enumerate(test_loader):
            for i in range(len(labels)):
                y = matrix.iter(labels[i])
                result = torch.cat((result,y))
        labels = result.detach().numpy()
        # max = labels.max()
        # min = labels.min()
        # for i in range(len(labels)):
            # labels[i] = (labels[i]-min)/(max-min)
        mean = np.mean(labels)
        std = np.std(labels)
        labels = (labels - mean) / std
        print("mean",mean)
        print("std",std)
        np.savetxt('./new_test_labels.csv', labels, fmt='%f')
        # return labels


def gen_result(time_matrix, labels, slidingWindow):
    res = torch.mm(time_matrix, labels.resize(slidingWindow,1).double())    
    return res

# def gen_label(slidingWindow, batch_size, reset=False):
#     if reset:
#         matrix = create_matrix(slidingWindow)   
#         train_loader, test_loader = get_raw_data(batch_size)
#         perior_result = torch.zeros([0,1])
#         targets = torch.zeros([slidingWindow, 1])
#         for batch_index, (_, labels) in enumerate(train_loader):
#             if batch_index == 0:
#                 targets = labels[0:slidingWindow]
#                 for i in range(len(labels)-slidingWindow):
#                     if i == 0:
#                         continue
#                     targets[:slidingWindow-1] = targets[1:slidingWindow].clone()
#                     targets[-1] = labels[i+slidingWindow-1]
#                     result = gen_result(matrix, targets, slidingWindow)
#                     result = result.float()
#                     perior_result = torch.cat((perior_result,result.view(1,1)))
#             else:
#                 for i in range(len(labels)):
#                     targets[:slidingWindow-1] = targets[1:slidingWindow].clone()
#                     targets[-1] = labels[i]
#                     result = gen_result(matrix, targets, slidingWindow)
#                     result = result.float()
#                     perior_result = torch.cat((perior_result,result.view(1,1)))
#         labels = perior_result.detach().numpy()
#         max = labels.max()
#         min = labels.min()
#         for i in range(len(labels)):
#             labels[i] = (labels[i]-min)/(max-min)
#         print("mean",np.mean(labels))
#         print("var",np.var(labels))
#         np.savetxt('./train_labels.csv', labels, fmt='%f')

#         perior_result = torch.zeros([0,1])
#         targets = torch.zeros([slidingWindow, 1])
#         for batch_index, (_, labels) in enumerate(test_loader):
#             if batch_index == 0:
#                 targets = labels[0:slidingWindow]
#                 for i in range(len(labels)-slidingWindow):
#                     if i == 0:
#                         continue
#                     targets[:slidingWindow-1] = targets[1:slidingWindow].clone()
#                     targets[-1] = labels[i+slidingWindow-1]
#                     result = gen_result(matrix, targets, slidingWindow)
#                     result = result.float()
#                     perior_result = torch.cat((perior_result,result.view(1,1)))
#             else:
#                 for i in range(len(labels)):
#                     targets[:slidingWindow-1] = targets[1:slidingWindow].clone()
#                     targets[-1] = labels[i]
#                     result = gen_result(matrix, targets, slidingWindow)
#                     result = result.float()
#                     perior_result = torch.cat((perior_result,result.view(1,1)))
#         labels = perior_result.detach().numpy()
#         max = labels.max()
#         min = labels.min()
#         for i in range(len(labels)):
#             labels[i] = (labels[i]-min)/(max-min)
#         np.savetxt('./test_labels.csv', labels, fmt='%f')

def load_label(train):
    if train:
        labels = np.loadtxt('./new_train_labels.csv')
    else:
        labels = np.loadtxt('./new_test_labels.csv')
    return labels

gen_label(50,1, 256, True)