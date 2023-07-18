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
        self.relu = nn.SELU()
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

        return im

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
        
        return im

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

        return im

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
        
        return im


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

        return im


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

    def fit(self,train_loader,valid_loader,n_epochs,device,criterion,optimizer,data_path,scheduler):

        model = self.to(device)
        tbw = SummaryWriter()

        rmse= RMSELoss()
        [mean_mod,std_mod] = torch.load(data_path+'mean_std_mod.pt')
        [mean_u,std_u] = torch.load(data_path+'mean_std_u.pt')
        [mean_v,std_v] = torch.load(data_path+'mean_std_v.pt')
    

        for epoch in range(n_epochs):
            print('epoch: {}'.format(epoch+1))

            l_loss1 = []
            l_loss2 = []
            l_loss3 = []
            l_loss4 = []
            
            for i,(ssh3,ssh6,ssh12,sst6,sst12,u,v) in tqdm(enumerate(train_loader)):

                optimizer.zero_grad()

                X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y = [ssh6.to(device),ssh12.to(device),u.to(device),v.to(device)]

                b_size = ssh3.shape[0]
                if b_size == 0:
                    continue

                y_pred = model(X)

                loss1 = criterion(y_pred[0],y[0])
                loss2 = criterion(y_pred[1],y[1])
                loss3 = criterion(y_pred[2],y[2])
                loss4 = criterion(y_pred[3],y[3])
                loss = loss1 + loss2 + loss3 + loss4

                l_loss1.append(loss1.item())
                l_loss2.append(loss2.item())
                l_loss3.append(loss3.item())
                l_loss4.append(loss4.item())

                loss.backward()

                optimizer.step()


            l_valid1 = []
            l_valid2 = []
            l_valid3 = []

            with torch.no_grad():

                for i,(ssh3,ssh6,ssh12,sst6,sst12,u,v) in tqdm(enumerate(valid_loader)):

                    optimizer.zero_grad()

                    X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                    y = [ssh6.to(device),ssh12.to(device),u.to(device),v.to(device)]

                    b_size = ssh3.shape[0]
                    if b_size == 0:
                        continue

                    y_pred = model(X)
             
                    l_valid1.append(rmse(y_pred[1]*std_mod+mean_mod,y[1]*std_mod+mean_mod).item())
                    l_valid2.append(rmse(y_pred[2]*std_u+mean_u,y[2]*std_u+mean_u).item())
                    l_valid3.append(rmse(y_pred[3]*std_v+mean_v,y[3]*std_v+mean_v).item())

                valid_mean1 = np.array(l_valid1).mean()
                tbw.add_scalar("RMSE SLA 1/12(m)",valid_mean1,epoch)

                valid_mean2 = np.array(l_valid2).mean()
                tbw.add_scalar("RMSE U (m.s-1)",valid_mean2,epoch)

                valid_mean3 = np.array(l_valid3).mean()
                tbw.add_scalar("RMSE V (m.s-1)",valid_mean3,epoch)

                print('RMSE: {}m'.format(valid_mean1))


            tbw.add_scalar("loss 1",np.array(l_loss1).mean(),epoch)
            tbw.add_scalar("loss 2",np.array(l_loss2).mean(),epoch)
            tbw.add_scalar("loss 3",np.array(l_loss3).mean(),epoch)
            tbw.add_scalar("loss 4",np.array(l_loss4).mean(),epoch)

            if epoch%4 == 0 and i==10:
                tbw.add_image("prediction ssh12",y_pred[1][0])
                tbw.add_image("target ssh",y[1][0])
                tbw.add_image("prediction u",y_pred[2][0])
                tbw.add_image("target u",y[2][0])


        scheduler.step(valid_mean1)

        print('lr:{}'.format(optimizer.param_groups[0]["lr"]))


        tbw.close()
            

    def test(self,criterion,test_loader,device,data_path,get_im):

        [mean_mod,std_mod] = torch.load(data_path+'mean_std_mod.pt')
        [mean_u,std_u] = torch.load(data_path+'mean_std_u.pt')
        [mean_v,std_v] = torch.load(data_path+'mean_std_v.pt')
        model = self.to(device)
        test_accuracy = [] 
        test_accuracy2 = [] 
        test_accuracy3 = []    
 
        l_im = []
        with torch.no_grad():
            for i,(ssh3,ssh6,ssh12,sst6,sst12,u,v) in enumerate(test_loader):
                X = [ssh3.to(device),sst6.to(device),sst12.to(device)]
                y = [ssh6.to(device),ssh12.to(device),u.to(device),v.to(device)]

                y_pred = model(X)

                test_accuracy.append(criterion(y_pred[1]*std_mod+mean_mod,y[1]*std_mod+mean_mod).item())
                test_accuracy2.append(criterion(y_pred[2]*std_u+mean_u,y[2]*std_u+mean_u).item())
                test_accuracy3.append(criterion(y_pred[3]*std_v+mean_v,y[3]*std_v+mean_v).item())


                if i in get_im:
                    l_im.append([ssh3*std_mod+mean_mod,y_pred[1]*std_mod+mean_mod,y_pred[2]*std_u+mean_u,y_pred[3]*std_v+mean_v,y[1]*std_mod+mean_mod,y[2]*std_u+mean_u,y[3]*std_v+mean_v])


        test_accuracy = np.array(test_accuracy)
        test_accuracy2 = np.array(test_accuracy2)
        test_accuracy3 = np.array(test_accuracy3)
        mean = np.mean(test_accuracy, axis=0)
        mean2 = np.mean(test_accuracy2, axis=0)
        mean3 = np.mean(test_accuracy3, axis=0)

        if len(get_im)!=0:
            return mean,mean2,mean3,l_im
        else:
            return mean,mean2,mean3








class RMSELoss(torch.nn.Module):
    def __init__(self,coeff=1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.coeff = coeff
        
    def forward(self,yhat,y):
        return self.coeff*torch.sqrt(self.mse(yhat,y))



 






        