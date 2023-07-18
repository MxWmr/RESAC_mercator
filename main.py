
import torch
from datetime import datetime
from data_load import *
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
data_path = "/usr/home/mwemaere/neuro/Data3/"

pool = torch.nn.AvgPool2d(2,stride=(2,2))
pool2 = torch.nn.AvgPool2d(4,stride=(4,4))
train_loader = Dataset(98,96,data_path,'ssh_mod_','sst_','u_','v_',batch_size=16,first_file=0) # first file 67 to start with 2011


test_sla_12 = torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:264]
test_sst_12 = torch.load(data_path + 'test_sst.pt')[:,:,:,:264]

test_sla_6 = pool(torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:264])
test_sst_6 = pool(torch.load(data_path + 'test_sst.pt')[:,:,:,:264])

test_sla_3 = pool2(torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:264])

test_u = torch.load(data_path + 'test_u.pt')[:,:,:,:264]
test_v = torch.load(data_path + 'test_v.pt')[:,:,:,:264]


test_loader = ConcatData([test_sla_3,test_sla_6,test_sla_12,test_sst_6,test_sst_12,test_u,test_v],shuffle=False)



valid_sla_12 = torch.load(data_path + 'valid_ssh_mod.pt')[:,:,:,:264]
valid_sst_12 = torch.load(data_path + 'valid_sst.pt')[:,:,:,:264]

valid_sla_6 = pool(torch.load(data_path + 'valid_ssh_mod.pt')[:,:,:,:264])
valid_sst_6 = pool(torch.load(data_path + 'valid_sst.pt')[:,:,:,:264])

valid_sla_3 = pool2(torch.load(data_path + 'valid_ssh_mod.pt')[:,:,:,:264])

valid_u = torch.load(data_path + 'valid_u.pt')[:,:,:,:264]
valid_v = torch.load(data_path + 'valid_v.pt')[:,:,:,:264]

valid_loader = ConcatData([valid_sla_3,valid_sla_6,valid_sla_12,valid_sst_6,valid_sst_12,valid_u,valid_v],shuffle=False)




criterion = RMSELoss()

model = resac()

if False:    #train

    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=custom_scheduler)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=0.1,patience=5)
    n_epochs = 100

    model.fit(train_loader,valid_loader,n_epochs,device,criterion,optim,data_path,scheduler)

    torch.save(model.state_dict(), save_path+date+'resac.pth')

if True:   #test
    device= 'cpu'
    date = '07_18_11:26_'
    model.load_state_dict(torch.load(save_path+date+'resac.pth'))
    model = model.to(device)

    mean,mean2,mean3,l_im = model.test(criterion,test_loader,device,data_path, get_im=[291])


    print('test RMSE SLA 1/12:{}'.format(mean))
    print('test RMSE U:{}'.format(mean2))
    print('test RMSE V:{}'.format(mean3))



    plot_test_ssh(l_im,save_path,date)
    plot_test_uv(l_im,save_path,date)
    plot_test_uv2(l_im,save_path,date)