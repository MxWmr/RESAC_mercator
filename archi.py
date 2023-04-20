import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class resac(nn.Module):
    def __init__(self):
        super().__init__()
        
        nf = 36

        self.conv1 = nn.Conv2d(2,nf,3,padding='same')
        self.conv2 = nn.Conv2d(nf,nf,3,padding='same')

        self.conv3 = nn.Conv2d(nf,nf,3,padding='same')
        self.conv4 = nn.Conv2d(nf,nf,3,padding='same')

        self.conv5 = nn.Conv2d(nf,nf,3,padding='same')
        self.conv6 = nn.Conv2d(nf,nf,3,padding='same')

        self.conv7 = nn.Conv2d(nf,nf,3,padding='same')
        self.conv8 = nn.Conv2d(nf,nf,3,padding='same')

        self.conv9 = nn.Conv2d(nf,nf,3,padding='same')
        self.conv10 = nn.Conv2d(nf,1,1,padding='same')

        nf = 24

        self.conv11 = nn.Conv2d(2,nf,5,padding='same')
        self.conv12 = nn.Conv2d(nf,nf,5,padding='same')

        self.conv13 = nn.Conv2d(nf,nf,5,padding='same')
        self.conv14 = nn.Conv2d(nf,nf,5,padding='same')

        self.conv15 = nn.Conv2d(nf,nf,5,padding='same')
        self.conv16 = nn.Conv2d(nf,nf,5,padding='same')

        self.conv17 = nn.Conv2d(nf,nf,5,padding='same')
        self.conv18 = nn.Conv2d(nf,1,1,padding='same')


        self.upsamp = nn.Upsample(scale_factor=2,mode='bicubic')
        self.bn1 = nn.BatchNorm2d(36)
        self.bn2 =  nn.BatchNorm2d(nf) 
        self.factiv = nn.ReLU()
        self.sig = nn.Sigmoid()

    def CNN1(self,im):
        im = self.factiv(self.conv1(im))
        im = self.factiv(self.conv2(im))
        im = self.bn1(im)

        im = self.factiv(self.conv3(im))
        im = self.factiv(self.conv4(im))
        im = self.bn1(im)

        im = self.factiv(self.conv5(im))
        im = self.factiv(self.conv6(im))
        im = self.bn1(im)

        im = self.factiv(self.conv7(im))
        im = self.factiv(self.conv8(im))
        im = self.bn1(im)

        im = self.factiv(self.conv9(im))
        ssh6 = self.sig(self.conv10(im))
        return ssh6

    def CNN2(self,im):
        im = self.factiv(self.conv11(im))
        im = self.factiv(self.conv12(im))
        im = self.bn2(im)

        im = self.factiv(self.conv13(im))
        im = self.factiv(self.conv14(im))
        im = self.bn2(im)

        im = self.factiv(self.conv15(im))
        im = self.factiv(self.conv16(im))
        im = self.bn2(im)

        im = self.factiv(self.conv17(im))
        ssh12 = self.sig(self.conv18(im))

        return ssh12

    def forward(self,X):
        ssh3,sst6,sst12 = X[0],X[1],X[2]
        ssh3_up = self.upsamp(ssh3)
        ssh_sst6 = torch.concat((ssh3_up,sst6),axis=1)
        ssh6 = self.CNN1(ssh_sst6)

        ssh6_up = self.upsamp(ssh6)
        ssh_sst12 = torch.concat((ssh6_up,sst12),axis=1)
        ssh12 = self.CNN2(ssh_sst12)

        return [ssh6,ssh12]

    def fit(self,device,optimizer,criterion,train_loader,num_epochs):

        model = self.to(device)
        tbw = SummaryWriter()

        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch+1))

            l_loss1 = []
            l_loss2 = []
            
            for (ssh3,ssh6,ssh12,sst6,sst12) in tqdm(train_loader):

                optimizer.zero_grad()

                X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y = [ssh6.to(device),ssh12.to(device)]

                y_pred = model(X)

                loss1 = criterion(y_pred[0],y[0])
                loss2 = criterion(y_pred[1],y[1])
                loss = loss1 + loss2
                l_loss1.append(loss1.item())
                l_loss2.append(loss2.item())

                loss.backward()

                optimizer.step()

            tbw.add_scalar("loss 1",np.array(l_loss1).mean(),epoch)
            tbw.add_scalar("loss 2",np.array(l_loss2).mean(),epoch)

        tbw.close()
            

    def test(self,criterion,test_loader,device,get_im):
        model = self.to(device)
        test_accuracy = []    
        l_im = []
        with torch.no_grad():
            for i,(ssh3,ssh6,ssh12,sst6,sst12) in enumerate(test_loader):
                X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y = [ssh6.to(device),ssh12.to(device)]
                y_pred = model(X)
                test_accuracy.append(criterion(y[0],y_pred[0]).item()+criterion(y[1],y_pred[1]).item())

                if i in get_im:
                    l_im.append([ssh3,y_pred[0],y_pred[1],y[1]])


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

def prepare_loaders(ssh3,ssh6,ssh12,sst6,sst12,batch_size=32):

    l = [ssh3,ssh6,ssh12,sst6,sst12]
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





 






        