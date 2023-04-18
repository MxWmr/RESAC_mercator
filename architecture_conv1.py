import torch
import math as mt
import numpy as np
from tqdm import tqdm
import optuna
from torch.utils.tensorboard import SummaryWriter
from plot_fct_conv1 import *






class RESAC_MERCATOR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamp = torch.nn.Upsample(scale_factor=2,mode='bicubic')

        n_f  = 64

        l_conv,l2_conv = [],[]

        l_conv.append(torch.nn.Conv2d(2,n_f,3,padding='same'))
        l2_conv.append(torch.nn.Conv2d(n_f,n_f,3,padding='same'))
        for i in range(5):  
            l_conv.append(torch.nn.Conv2d(n_f,n_f,3,padding='same'))
            l2_conv.append(torch.nn.Conv2d(n_f,n_f,3,padding='same'))

        l2_conv.append(torch.nn.Conv2d(n_f,n_f,3,padding='same'))
        l_conv.append(torch.nn.Conv2d(n_f,1,3,padding='same'))

        self.l_conv = torch.nn.ModuleList(l_conv)
        self.l2_conv = torch.nn.ModuleList(l2_conv)

        self.bn = torch.nn.BatchNorm2d(n_f)
        self.relu = torch.nn.SiLU()
        self.sig = torch.nn.Sigmoid()



    def CNN1(self,im): 
        im = self.l_conv[0](im)
        im = self.relu(im)
        im = self.l2_conv[0](im)
        im = self.relu(im)
        im = self.bn(im)
        for i in range(1,len(self.l_conv)-1):
            im = self.l_conv[i](im)
            im = self.relu(im)
            im = self.l2_conv[i](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l2_conv[-1](im)
        im = self.relu(im)
        im = self.l_conv[-1](im)
        im =self.sig(im)


        return im
    


    def forward(self,X,up=False):
        ssh3,sst6 = X[0],X[1]

        ssh3_up = self.upsamp(ssh3)

        ssh_sst_6 = torch.concat((ssh3_up,sst6),axis=1)

        ssh6 = self.CNN1(ssh_sst_6) 

        if up:
            return ssh6,ssh3_up
        else:
            return ssh6



def get_accuracy(x,y):
    f_acc = torch.nn.MSELoss()
    acc_ssh6 = torch.sqrt(f_acc(x,y))

    return acc_ssh6.item()


def train_resac(model, device, optimizer, criterion, train_loader,valid_loader, num_epochs, verbose=1, scheduler=False, trial=False,tb=False):
    if tb:
        tbw = SummaryWriter()
    model = model.to(device)

    valid_accuracy=[]


    for epoch in range(num_epochs):
        if verbose:
            print('epoch: {}'.format(epoch+1))
        total_loss = 0
        for (ssh3,ssh6,sst6) in train_loader:

            #clear out the gradients from the last step loss.backward()
            optimizer.zero_grad()
            
            # cast data to cuda
            X_batch = [ssh3.to(device),sst6.to(device)]
            y_batch = ssh6.to(device)

            #forward feed
            y_pred = model(X_batch)


            #calculate the loss
            loss = criterion(y_pred, y_batch) 
            total_loss += loss.item()
            #backward propagation: calculate gradients
            loss.backward()
            
            #update the weights
            optimizer.step()
            
        if scheduler!=False:
            scheduler.step()

        with torch.no_grad():
            l_valid = []
            for (ssh3,ssh6,sst6) in valid_loader:
                X_valid = [ssh3.to(device),sst6.to(device)]
                y_valid = ssh6.to(device)
                output_valid = model(X_valid)
                l_valid.append(get_accuracy(output_valid, y_valid))
            l_valid = np.array(l_valid)
            valid_accuracy.append(np.mean(l_valid,axis=0))

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}, valid Accuracy: {valid_accuracy[-1]}")

        if tb:

            tbw.add_scalar("Validation loss",valid_accuracy[-1],epoch)
            tbw.add_scalar("Total loss",total_loss,epoch)

            if epoch % 5 == 0 :
                tbw.add_image("prediction",y_pred[0])
                tbw.add_image("target",y_batch[0])


        if trial != False:
            trial.report(valid_accuracy[-1], epoch)


            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    tbw.close()
    return valid_accuracy

def test_resac(model,test_loader,device,get_im=[]):

    test_accuracy = []    
    l_im = []
    with torch.no_grad():
        for i,(ssh3,ssh6,sst6) in enumerate(test_loader):
            X = [ssh3.to(device),sst6.to(device)]
            y = ssh6.to(device)
            y_pred,up = model(X,up=True)
            test_accuracy.append(get_accuracy(y_pred,y))

            if i in get_im:
                l_im.append([[up,sst6],y,y_pred])

    test_accuracy = np.array(test_accuracy)
    mean = np.mean(test_accuracy, axis=0)
    std = np.std(test_accuracy, axis=0)
    if len(get_im)!=0:
        return mean,std,l_im
    else:
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


def prepare_loaders(ssh3,ssh6,sst6,batch_size=32):


    l = [ssh3,ssh6,sst6]
    l_test = []


    # year 2005 for testing
    for i in range(len(l)):
        l_test.append(l[i][365*12+3:365*13+3,:,:,:])
        l[i] = torch.concat((l[i][:365*12+3,:,:,:],l[i][365*13+3:,:,:,:]),axis=0)

 
    test_loader = ConcatDataset(l_test,shuffle=True)


    partition = [8000,1496]
    
    # random validation and test split
    generator2 = torch.Generator().manual_seed(36)
    train_ssh3,valid_ssh3, = torch.utils.data.random_split(l[0],partition,generator=generator2)
    train_ssh6,valid_ssh6 = torch.utils.data.random_split(l[1],partition,generator=generator2)
    train_sst6,valid_sst6 = torch.utils.data.random_split(l[2],partition,generator=generator2)


    # prepare loaders  
    train_loader = ConcatDataset([train_ssh3,train_ssh6,train_sst6],shuffle=True,batch_size=batch_size)

    valid_loader = ConcatDataset([valid_ssh3,valid_ssh6,valid_sst6],shuffle=True,batch_size=batch_size)

    return train_loader,valid_loader,test_loader

class RMSELoss(torch.nn.Module):
    def __init__(self,coeff=1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.coeff = coeff
        
    def forward(self,yhat,y):
        return self.coeff*torch.sqrt(self.mse(yhat,y))


def custom_scheduler(epoch):
    if epoch<20:
        return 2e-3
    elif epoch<60:
        return 2e-3*(mt.exp(-0.02)**(epoch-20))
    elif epoch<100:
        return 2e-3*mt.exp(-0.02*40)*(mt.exp(-0.05)**(epoch-60))
    else:
        return 2e-3*mt.exp(-0.02*40)*mt.exp(-0.05*40)

def custom_scheduler3(epoch):
    if epoch<8:
        return 1
    elif epoch<17:
        return 0.5
    elif epoch<28:
        return 0.2
    elif epoch<38:
        return 0.1
    elif epoch<48:
        return 0.02
    else:
        return 0.01


