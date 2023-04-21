
import torch
from datetime import datetime
import numpy as np 
from archi import *
from plot_utils import *



if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

# load data
save_path = "/usr/home/mwemaere/neuro/resac_mercator/Save/"
data_path = "/usr/home/mwemaere/neuro/Data/"
ssh3 = torch.load(data_path + "SSH_MERCATOR_1%3.pt")
ssh6 = torch.load(data_path + "SSH_MERCATOR_1%6.pt")[:,:,:134]
ssh12 = torch.load(data_path + "SSH_MERCATOR_1%12.pt")[:,:,:268]
sst6 = torch.load(data_path + "SST_MERCATOR_1%6.pt")[:,:,:134]
sst12 = torch.load(data_path + "SST_MERCATOR_1%12.pt")[:,:,:268]
u12 = torch.load(data_path + "U_MERCATOR_1%12.pt")[:,:,:268]
v12 = torch.load(data_path + "V_MERCATOR_1%12.pt")[:,:,:268]


for A in (ssh3,ssh6,ssh12,sst6,sst12,u12,v12):
    A -= torch.min(A)
    A /= torch.max(A)


ssh3 = torch.unsqueeze(ssh3,1)
ssh6 = torch.unsqueeze(ssh6,1)
ssh12 = torch.unsqueeze(ssh12,1)
sst6 = torch.unsqueeze(sst6,1)
sst12 = torch.unsqueeze(sst12,1)
u12 = torch.unsqueeze(u12,1)
v12 = torch.unsqueeze(v12,1)

train_loader,test_loader = prepare_loaders(ssh3,ssh6,ssh12,sst6,sst12,u12,v12,batch_size=16)

criterion = RMSELoss()

model = resac()

if True:
    lr = 1e-3  
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    
    num_epochs = 70

    model.fit(device,optimizer,criterion,train_loader,num_epochs)

    torch.save(model.state_dict(), save_path+date+'model_conv1.pth')

if False:
    device= 'cpu'
    date = '04_20_15:21_'
    model.load_state_dict(torch.load(save_path+date+'model_conv1.pth'))
    model = model.to(device)

    mean,std, l_im = model.test(criterion,test_loader,device, get_im=[15,58,245])


    print(mean)
    print(std)


    plot_test_ssh(l_im,save_path,date)
    plot_test_uv(l_im,save_path,date)