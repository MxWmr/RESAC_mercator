
# import modules
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture import *
from plot_fct import *



if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

# load data

data_path = "/usr/home/mwemaere/neuro/Data/"
ssh3 = torch.load(data_path + "SSH_MERCATOR_1%3.pt")
ssh6 = torch.load(data_path + "SSH_MERCATOR_1%6.pt")[:,:,:134]
ssh12 = torch.load(data_path + "SSH_MERCATOR_1%12.pt")[:,:,:268]
sst6 = torch.load(data_path + "SST_MERCATOR_1%6.pt")[:,:,:134]
sst12 = torch.load(data_path + "SST_MERCATOR_1%12.pt")[:,:,:268]
u12 = torch.load(data_path + "U_MERCATOR_1%12.pt")[:,:,:268]
v12 = torch.load(data_path + "V_MERCATOR_1%12.pt")[:,:,:268]


ssh3 = torch.unsqueeze(ssh3,1)
ssh6 = torch.unsqueeze(ssh6,1)
ssh12 = torch.unsqueeze(ssh12,1)
sst6 = torch.unsqueeze(sst6,1)
sst12 = torch.unsqueeze(sst12,1)
u12 = torch.unsqueeze(u12,1)
v12 = torch.unsqueeze(v12,1)

# prepare data
train_loader,valid_loader,test_loader = prepare_loaders(ssh3,ssh6,ssh12,sst6,sst12,u12,v12)

# create model
model = RESAC_MERCATOR()


saved_path = '/usr/home/mwemaere/neuro/resac_mercator/Save/04_11_18:16_model.pth'
model.load_state_dict(torch.load(saved_path))
model = model.to(device)



# Test
mean,std, l_im = test_resac(model,test_loader,device, get_im=[15,58,245])


print(mean)
print(std)

im1 = torch.squeeze(l_im[0][1][1]).cpu().numpy()    # true
im2 = torch.squeeze(l_im[0][2][1]).cpu().numpy()    # pred


print("mean true im :{}  mean pred: {}".format(np.mean(im1),np.mean(im2)))

print("std true im :{}  std pred: {}".format(np.std(im1),np.std(im2)))

print("min val true im :{}  min val pred: {}".format(np.amin(im1),np.amin(im2)))

print("max val true im :{}  max val pred: {}".format(np.amax(im1),np.amax(im2)))


# plot_test_uv(l_im)
# plot_test_ssh(l_im)