import copy

import torch
from model import resnet
from data.dataset import DogCat
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

max_epoch = 128
batch_size = 32
lr = 0.00005
weight_decay = 0
USE_CUDA = torch.cuda.is_available()


#开始训练模型
def train():
    net = resnet.net
    #数据部分：
    train_data = DogCat('data/train',train=True)
    val_data = DogCat(root='data/train')
    train_loader = DataLoader( train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True)

    #定义损失函数以及
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)

    #开始训练
    best_acc = 0
    for epoch in range(max_epoch):
        running_corrects = 0
        for i ,(data,label) in enumerate(train_loader):
            input = data
            target = label
            if USE_CUDA:
                input = input.cuda()
                target = target.cuda()
                net = net.cuda()
            predict = net(input)
            _,pre = torch.max(predict,1)
            loss = criterion(predict,target)
            loss.backward()
            optimizer.step()
            print(f'epoch:{epoch},batch={i},loss={loss}')
            running_corrects += torch.sum(pre == target)
        epoch_acc = running_corrects.double() / len(train_data)
        print(epoch_acc)

        if(epoch_acc > best_acc):
            best_acc = epoch_acc
            best_model = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_model)
    torch.save(net.state_dict(),'model.pth')



def test():
    net = resnet.net
    net.load_state_dict(torch.load('model.pth'))
    #测试数据
    test_data = DogCat('data/test',test=True)
    test_load = DataLoader(test_data,batch_size = batch_size,shuffle=False)

    #结果保存
    df = pd.DataFrame()
    id = []
    label = []

    for i , (data,path) in enumerate(test_load):
        id.append(path.numpy())
        input = data
        if USE_CUDA:
            input = input.cuda()
        predict = net(input)
        _,pre = torch.max(predict,1)
        if pre == 1:
            label.append('dog')
        else:
            label.append('cat')

    dict = {'id':id,'label':label}
    df.to_csv('result/result.csv')


if __name__ == '__main__':
    train()
    test()


