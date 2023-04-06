import torch
import math as mt
import numpy as np
from tqdm import tqdm

class RESAC_MERCATOR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamp = torch.nn.Upsample(scale_factor=2,mode='nearest')

        l_conv = []
        l_conv.append(torch.nn.Conv2d(2,36,(6,6),padding='same'))
        for i in range(4):
            l_conv.append(torch.nn.Conv2d(36,36,(6,6),padding='same'))
        l_conv.append(torch.nn.Conv2d(36,1,(1,1)))
        self.l_conv = torch.nn.ModuleList(l_conv)

        self.bn = torch.nn.BatchNorm2d(36)
        self.relu = torch.nn.ReLU()
        self.sig = torch.sigmoid

        l_conv2 = []
        l_conv2.append(torch.nn.Conv2d(2,24,(5,5),padding='same'))
        l_conv2.append(torch.nn.Conv2d(24,24,(5,5),padding='same'))
        l_conv2.append(torch.nn.Conv2d(24,1,(1,1),padding='same'))
        self.l_conv2 = torch.nn.ModuleList(l_conv2)

        self.bn2 = torch.nn.BatchNorm2d(24)

        l_conv3 = []
        l_conv3.append(torch.nn.Conv2d(2,24,(5,5),padding='same'))
        l_conv3.append(torch.nn.Conv2d(24,24,(5,5),padding='same'))
        l_conv3.append(torch.nn.Conv2d(24,1,(1,1),padding='same'))
        self.l_conv3 = torch.nn.ModuleList(l_conv3)      

        l_conv4 = []
        l_conv4.append(torch.nn.Conv2d(1,24,(5,5),padding='same'))
        l_conv4.append(torch.nn.Conv2d(24,24,(5,5),padding='same'))
        l_conv4.append(torch.nn.Conv2d(24,1,(1,1),padding='same'))
        self.l_conv4 = torch.nn.MoannulerduleList(l_conv4)  

        l_conv5 = []
        l_conv5.append(torch.nn.Conv2d(1,24,(5,5),padding='same'))
        l_conv5.append(torch.nn.Conv2d(24,24,(5,5),padding='same'))
        l_conv5.append(torch.nn.Conv2d(24,1,(1,1),padding='same'))
        self.l_conv5 = torch.nn.ModuleList(l_conv5)  

    def CNN1(self,im):
        for i in range(len(self.l_conv)-1):
            im = self.l_conv[i](im)
            im = self.relu(im)
            im = self.bn(im)
        im = self.l_conv[-1](im)
        im = self.sig(im)
        return im
    
    def CNN2(self,im):
        for i in range(len(self.l_conv3)-1):
            im = self.l_conv3[i](im)
            im = self.relu(im)
            im = self.bn2(im)
        im = self.l_conv3[-1](im)
        im = self.sig(im)
        return im
    
    def CNN3(self,im):
        for i in range(len(self.l_conv3)-1):
            im = self.l_conv3[i](im)
            im = self.relu(im)
            im = self.bn2(im)
        im = self.l_conv3[-1](im)
        im = self.sig(im)
        return im
    
    def CNN4(self,im):
        for i in range(len(self.l_conv4)-1):
            im = self.l_conv4[i](im)
            im = self.relu(im)
            im = self.bn2(im)
        im = self.l_conv4[-1](im)
        im = self.sig(im)
        return im

    def CNN5(self,im):
        for i in range(len(self.l_conv5)-1):
            im = self.l_conv5[i](im)
            im = self.relu(im)
            im = self.bn2(im)
        im = self.l_conv5[-1](im)
        im = self.sig(im)
        return im
    

    def forward(self,X):
        ssh3,sst6,sst12 = X[0],X[1],X[2]

        ssh3_up = self.upsamp(ssh3)
        ssh_sst_6 = torch.concat((ssh3_up,sst6),axis=1)
        ssh6 = self.CNN1(ssh_sst_6)


        ssh6_up = self.upsamp(ssh6)
        ssh_sst_12 = torch.concat((ssh6_up,sst12),axis=1)
        ssh12 = self.CNN2(ssh_sst_12)

        ssh_sst_12_bis = torch.concat((ssh12,sst12),axis=1)
        uv_12 = self.CNN3(ssh_sst_12_bis)

        u = self.CNN4(uv_12)
        v = self.CNN5(uv_12)

        y = [ssh6,ssh12,u,v]

        return y




def get_accuracy(x,y):
    f_acc = torch.nn.MSELoss()
    acc_ssh6 = torch.sqrt(f_acc(x[0],y[0]))
    acc_ssh12 = torch.sqrt(f_acc(x[1],y[1]))
    acc_u = torch.sqrt(f_acc(x[2],y[2]))
    acc_v = torch.sqrt(f_acc(x[3],y[3]))
    return [acc_ssh6.item(),acc_ssh12.item(),acc_u.item(),acc_v.item()]


def train_resac(model, device, optimizer, scheduler, criterion, train_loader,valid_loader, num_epochs):
    model = model.to(device)
    train_accuracy=[]
    valid_accuracy=[]


    for epoch in range(num_epochs):
        print('epoch: {}'.format(epoch+1))
        for (ssh3,ssh6,ssh12,sst6,sst12,u12,v12) in tqdm(train_loader):

            X_batch = [ssh3.to(device),sst6.to(device),sst12.to(device)]
            y_batch = [ssh6.to(device),ssh12.to(device),u12.to(device),v12.to(device)]
            #forward feed
            y_pred= model(X_batch)


            #calculate the loss
            loss1 = criterion(y_pred[0], y_batch[0])  # ssh6
            loss2 = criterion(y_pred[1], y_batch[1])   # ssh12
            loss3 = criterion(y_pred[2], y_batch[2])  # u
            loss4 = criterion(y_pred[3], y_batch[3])  # v
            loss = loss1 + loss2 + loss3 + loss4

            train_accuracy.append([loss1.item(),loss2.item(),loss3.item(),loss4.item()])

            #clear out the gradients from the last step loss.backward()
            optimizer.zero_grad()
            
            #backward propagation: calculate gradients
            loss.backward()
            
            #update the weights
            optimizer.step()
            
        scheduler.step()

        with torch.no_grad():

            for (ssh3,ssh6,ssh12,sst6,sst12,u12,v12) in valid_loader:
                X_valid = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y_valid = [ssh6.to(device),ssh12.to(device),u12.to(device),v12.to(device)]
                output_valid = model(X_valid)
                valid_accuracy.append(get_accuracy(output_valid, y_valid))


        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy[-1]}, valid Accuracy: {valid_accuracy[-1]}")

    return train_accuracy, valid_accuracy

def test_resac(model,test_loader,device):

    test_accuracy = []    
    with torch.no_grad():
        for (ssh3,ssh6,ssh12,sst6,sst12,u12,v12) in test_loader:
            X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
            y = [ssh6.to(device),ssh12.to(device),u12.to(device),v12.to(device)]
            y_pred= model(X)
            test_accuracy.append(get_accuracy(y_pred,y))

    test_accuracy = np.array(test_accuracy)
    mean = np.mean(test_accuracy, axis=0)
    std = np.std(test_accuracy, axis=0)
    return mean,std


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self,datasets,shuffle=False,batch_size=1):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size

        if shuffle:
            n  = len(datasets[0])
            id_rd = torch.randperm(n)
            for d in self.datasets:
                d = d[list(id_rd)]

    def __getitem__(self,i):
        self.datasets[0][(i+1)*self.batch_size]
        return tuple(d[i*self.batch_size:(i+1)*self.batch_size] for d in self.datasets)

    def __len__(self):
        return min(int(len(d)/self.batch_size) for d in self.datasets)


def prepare_loaders(ssh3,ssh6,ssh12,sst6,sst12,u12,v12,batch_size=32):


    X = [ssh3,sst6,sst12]
    y = [ssh6,ssh12,u12,v12]
    X_test = []
    y_test = []

    # year 2005 for testing
    for i in range(len(X)):
        X_test.append(X[i][365*12+3:365*13+3,:,:,:])
        X[i] = torch.concat((X[i][:365*12+3,:,:,:],X[i][365*13+3:,:,:,:]),axis=0)

    for i in range(len(y)):
        y_test.append(y[i][365*12+3:365*13+3,:,:,:])
        y[i] = torch.concat((y[i][:365*12+3,:,:,:],y[i][365*13+3:,:,:,:]),axis=0)

    X_test.extend(y_test)
    test_loader = ConcatDataset(X_test,shuffle=True)


    partition = [9000,496]
    
    # random validation and test split
    generator2 = torch.Generator().manual_seed(42)
    train_ssh3,valid_ssh3, = torch.utils.data.random_split(X[0],partition,generator=generator2)
    train_ssh6,valid_ssh6 = torch.utils.data.random_split(y[0],partition,generator=generator2)
    train_ssh12,valid_ssh12 = torch.utils.data.random_split(y[1],partition,generator=generator2)
    train_sst6,valid_sst6 = torch.utils.data.random_split(X[1],partition,generator=generator2)
    train_sst12,valid_sst12 = torch.utils.data.random_split(X[2],partition,generator=generator2)
    train_u12,valid_u12 = torch.utils.data.random_split(y[2],partition,generator=generator2)
    train_v12,valid_v12 = torch.utils.data.random_split(y[3],partition,generator=generator2)


    # prepare loaders  
    train_loader = ConcatDataset([train_ssh3,train_ssh6,train_ssh12,train_sst6,train_sst12,train_u12,train_v12],shuffle=True,batch_size=batch_size)

    valid_loader = ConcatDataset([valid_ssh3,valid_ssh6,valid_ssh12,valid_sst6,valid_sst12,valid_u12,valid_v12],shuffle=False,batch_size=batch_size)

    return train_loader,valid_loader,test_loader

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


def custom_scheduler(epoch):
    if epoch<20:
        return 2e-3
    elif epoch<60:
        return 2e-3*(mt.exp(-0.02)**(epoch-20))
    elif epoch<100:
        return 2e-3*mt.exp(-0.02*40)*(mt.exp(-0.05)**(epoch-60))
    else:
        return 2e-3*mt.exp(-0.02*40)*mt.exp(-0.05*40)


