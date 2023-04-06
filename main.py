
# import modules
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture import *



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

# training 

optimizer = torch.optim.Adam(model.parameters(),lr=1.5e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, custom_scheduler)
criterion = RMSELoss()
num_epochs = 50
if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')

train_accuracy, valid_accuracy = train_resac(model, device, optimizer, scheduler, criterion, train_loader,valid_loader, num_epochs)


# save model weights
save_path = "/usr/home/mwemaere/neuro/resac_mercator/Save/"
date = datetime.now().strftime("%m_%d_%H:%M_")
torch.save(model.state_dict(), save_path+date+'model.pth')

# display loss
plt.figure(1)
plt.plot(train_accuracy,label=['ssh6','ssh12','u','v'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig(save_path+date+'train_loss.png')
plt.show()

plt.figure(1)
plt.plot(valid_accuracy,label=['ssh6','ssh12','u','v'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig(save_path+date+'valid_loss.png')
plt.show()

# test 

mean,std = test_resac(model,test_loader,device)
print(mean)
print(std)
