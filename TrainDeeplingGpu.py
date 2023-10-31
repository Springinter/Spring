import torch
import torchvision
#准备数据集

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
# from module import *
import math
train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_data_size = len(test_data)
test_data_size = len(train_data)
print("测试数据集的长度：{}".format(test_data_size))
print("训练数据集的长度：{}".format(train_data_size))

#利用dataLoader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#创建网络模型

#搭建神经网络
import torch
from torch import nn

class TuDui(nn.Module):
    def __init__(self):
        super(TuDui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),
        )
    def forward(self,x):
            x = self.model(x)
            return x
# if __name__ == '__main__':
#         tudui = TuDui()
        # input = torch.ones((64,3,32,32))#torch.ones((64,3,32,32))表示batch_size=64,channel=3,高H=32,宽W=32
        # output = tudui(input)
        # print (output.shape)
tudui = TuDui()
if torch.cuda.is_available():
    tudui.cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0

#训练的轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("login_train")
start_time = time.time()
for i in range(epoch):
    print("-----第{}轮开始了-----".format(i+1))
    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs,target = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            target = target.cuda()
        outputs = tudui(imgs)

        #优化器模型
        loss = loss_fn(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：loss:{}".format(total_train_step))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imsgs,target = data
            if torch.cuda.is_available():
                imsgs = imsgs.cuda()
                target = target.cuda()
            outputs = tudui(imsgs)
            loss = loss_fn(outputs,target)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1)==target).sum()
            total_accuracy = total_accuracy + accuracy
        print("整体测试集上的loss:{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
        writer.add_scalar("test_loss",total_test_loss,total_test_step)
        writer.add_scalar("test_accuracy",total_accuracy/total_test_loss,total_test_step)
        total_test_step = total_test_step + 1
        #保存模型
        torch.save(tudui,"tudui_{}.path".format(i))
        print("模型已保存")
writer.close()