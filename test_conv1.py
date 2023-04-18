
# import modules
import torch

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture_conv1 import *
from plot_fct_conv1 import *



if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')



# load data

data_path = "/usr/home/mwemaere/neuro/Data/"
ssh3 = torch.load(data_path + "SSH_MERCATOR_1%3.pt")
ssh6 = torch.load(data_path + "SSH_MERCATOR_1%6.pt")[:,:,:134]
sst6 = torch.load(data_path + "SST_MERCATOR_1%6.pt")[:,:,:134]




for A in (ssh3,ssh6,sst6):
    A -= torch.min(A)
    A /= torch.max(A)

ssh3 = torch.unsqueeze(ssh3,1)
ssh6 = torch.unsqueeze(ssh6,1)
sst6 = torch.unsqueeze(sst6,1)

# prepare data
train_loader,valid_loader,test_loader = prepare_loaders(ssh3,ssh6,sst6)

# create model
model = RESAC_MERCATOR()

date = '04_17_16:15_'
saved_path = '/usr/home/mwemaere/neuro/resac_mercator/Save/'+date+'model_conv1.pth'
model.load_state_dict(torch.load(saved_path))
model = model.to(device)



# Test
mean,std, l_im = test_resac(model,test_loader,device, get_im=[15,58,245])


print(mean)
print(std)
with open('test_result.txt', 'a') as f:
    f.write('\n'+date+' conv1 '+'\n')
    f.write(str(mean)+'\n')
    f.write(str(std)+'\n')

    f.close()

plot_test_ssh(l_im,saved_path,date)