import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class resac(nn.Module):
    def __init__(self):
        super().__init__()
        
        nf = 36

        l_conv1 = []
        l_conv1.append(nn.Conv2d(2,nf,3,padding='same'))

        for i in range(8):
            l_conv1.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv1.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv1 = nn.ModuleList(l_conv1)



        nf = 24

        l_conv2 = []
        l_conv2.append(nn.Conv2d(2,nf,5,padding='same'))

        for i in range(6):
            l_conv2.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv2.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv2 = nn.ModuleList(l_conv2)




        l_conv3 = []
        l_conv3.append(nn.Conv2d(2,nf,3,padding='same'))

        for i in range(6):
            l_conv3.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv3.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv3 = nn.ModuleList(l_conv3)




        l_conv4 = []
        l_conv4.append(nn.Conv2d(1,nf,3,padding='same'))

        for i in range(6):
            l_conv4.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv4.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv4 = nn.ModuleList(l_conv4)



        l_conv5 = []
        l_conv5.append(nn.Conv2d(1,nf,3,padding='same'))

        for i in range(6):
            l_conv5.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv5.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv5 = nn.ModuleList(l_conv5)







        self.upsamp = nn.Upsample(scale_factor=2,mode='bicubic')
        self.bn1 = nn.BatchNorm2d(36)
        self.bn =  nn.BatchNorm2d(nf) 
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def CNN1(self,im):
        for i in range(0,len(self.l_conv1)-2,2):
            im = self.l_conv1[i](im)
            im = self.relu(im)
            im = self.l_conv1[i+1](im)
            im = self.relu(im)
            im = self.bn1(im)

        im = self.l_conv1[-2](im)
        im = self.relu(im)
        im = self.l_conv1[-1](im)
        ssh6 =self.sig(im)

        return ssh6

    def CNN2(self,im):

        for i in range(0,len(self.l_conv2)-2,2):
            im = self.l_conv2[i](im)
            im = self.relu(im)
            im = self.l_conv2[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv2[-2](im)
        im = self.relu(im)
        im = self.l_conv2[-1](im)
        ssh12 =self.sig(im)
        
        return ssh12

    def CNN3(self,im):

        for i in range(0,len(self.l_conv3)-2,2):
            im = self.l_conv3[i](im)
            im = self.relu(im)
            im = self.l_conv3[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv3[-2](im)
        im = self.relu(im)
        im = self.l_conv3[-1](im)
        uv =self.sig(im)

        return uv

    def CNN4(self,im):

        for i in range(0,len(self.l_conv4)-2,2):
            im = self.l_conv4[i](im)
            im = self.relu(im)
            im = self.l_conv4[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv4[-2](im)
        im = self.relu(im)
        im = self.l_conv4[-1](im)
        u =self.sig(im)
        
        return u


    def CNN5(self,im):

        for i in range(0,len(self.l_conv3)-2,2):
            im = self.l_conv5[i](im)
            im = self.relu(im)
            im = self.l_conv5[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv5[-2](im)
        im = self.relu(im)
        im = self.l_conv5[-1](im)
        v =self.sig(im)

        return v  


    def forward(self,X):
        ssh3,sst6,sst12 = X[0],X[1],X[2]
        ssh3_up = self.upsamp(ssh3)
        ssh_sst6 = torch.concat((ssh3_up,sst6),axis=1)
        ssh6 = self.CNN1(ssh_sst6)

        ssh6_up = self.upsamp(ssh6)
        ssh_sst12 = torch.concat((ssh6_up,sst12),axis=1)
        ssh12 = self.CNN2(ssh_sst12)


        ssh_sst_12_bis = torch.concat((ssh12,sst12),axis=1)
        uv_12 = self.CNN3(ssh_sst_12_bis)

        u = self.CNN4(uv_12)
        v = self.CNN5(uv_12)

        return [ssh6,ssh12,u,v]

    def fit(self,device,optimizer,criterion,train_loader,num_epochs):

        model = self.to(device)
        tbw = SummaryWriter()

        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch+1))

            l_loss1 = []
            l_loss2 = []
            l_loss3 = []
            l_loss4 = []
            
            for (ssh3,ssh6,ssh12,sst6,sst12,u,v) in tqdm(train_loader):

                optimizer.zero_grad()

                X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y = [ssh6.to(device),ssh12.to(device),u.to(device),v.to(device)]

                y_pred = model(X)

                loss1 = criterion(y_pred[0],y[0])
                loss2 = criterion(y_pred[1],y[1])
                loss3 = criterion(y_pred[2],y[3])
                loss4 = criterion(y_pred[3],y[3])
                loss = loss1 + loss2 + loss3 + loss4
                l_loss1.append(loss1.item())
                l_loss2.append(loss2.item())
                l_loss3.append(loss3.item())
                l_loss4.append(loss4.item())

                loss.backward()

                optimizer.step()

            tbw.add_scalar("loss 1",np.array(l_loss1).mean(),epoch)
            tbw.add_scalar("loss 2",np.array(l_loss2).mean(),epoch)
            tbw.add_scalar("loss 3",np.array(l_loss3).mean(),epoch)
            tbw.add_scalar("loss 4",np.array(l_loss4).mean(),epoch)

            if epoch%4 == 0:
                tbw.add_image("prediction ssh12",y_pred[1][0])
                tbw.add_image("target ssh",y[1][0])
                tbw.add_image("prediction u",y_pred[2][0])
                tbw.add_image("target u",y[2][0])

        tbw.close()
            

    def test(self,criterion,test_loader,device,get_im):
        model = self.to(device)
        test_accuracy = []    
        l_im = []
        with torch.no_grad():
            for i,(ssh3,ssh6,ssh12,sst6,sst12,u,v) in enumerate(test_loader):
                X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y = [ssh6.to(device),ssh12.to(device),u.to(device),v.to(device)]
                y_pred = model(X)
                test_accuracy.append(criterion(y[0],y_pred[0]).item()+criterion(y[1],y_pred[1]).item()+criterion(y[2],y_pred[2]).item()+criterion(y[3],y_pred[3]).item())

                if i in get_im:
                    l_im.append([ssh3,y_pred[0],y_pred[1],y_pred[2],y_pred[3],y[1],y[2],y[3]])


        test_accuracy = np.array(test_accuracy)
        mean = np.mean(test_accuracy, axis=0)
        std = np.std(test_accuracy, axis=0)
        if len(get_im)!=0:
            return mean,std,l_im
        else:
            return mean,std
        
class ConcatData(torch.utils.data.Dataset):
    def __init__(self,datasets,shuffle=False,batch_size=1):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size

        if shuffle:
            n = len(datasets[0])
            id_rd = torch.randperm(n)
            for d in self.datasets:
                d = d[list(id_rd)]

    def __getitem__(self,i):
        self.datasets[0][(i+1)*self.batch_size]
        return tuple(d[i*self.batch_size:(i+1)*self.batch_size] for d in self.datasets)


    def __len__(self):
        return min(int(len(d)/self.batch_size) for d in self.datasets)

def prepare_loaders(ssh3,ssh6,ssh12,sst6,sst12,u,v,batch_size=32):

    l = [ssh3,ssh6,ssh12,sst6,sst12,u,v]
    l_test = []

    for i in range(len(l)):
        l_test.append(l[i][365*12+3:365*13+3,:,:,:])
        l[i] = torch.concat((l[i][:365*12+3,:,:,:],l[i][365*13+3:,:,:,:]),axis=0)


    test_loader = ConcatData(l_test,shuffle=True)

    train_loader = ConcatData(l,shuffle=True,batch_size=batch_size)
    
    return train_loader,test_loader

class RMSELoss(torch.nn.Module):
    def __init__(self,coeff=1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.coeff = coeff
        
    def forward(self,yhat,y):
        return self.coeff*torch.sqrt(self.mse(yhat,y))





 






        