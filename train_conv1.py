
# import modules
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture_conv1 import *
from plot_fct_conv1 import *



if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

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


# prepare data= self.sig(im)
train_loader,valid_loader,test_loader = prepare_loaders(ssh3,ssh6,sst6,batch_size=16)

# create model
model = RESAC_MERCATOR()





# training 
lr = 1e-3   
#lambda1 = lambda epoch: 0.65 ** epoch
def lambda1(epoch):
    if epoch<20:
        return 0.99**epoch
    elif epoch<70:
        return 0.96**epoch
    else:
        return 0.94**epoch

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
criterion = RMSELoss()
#criterion = torch.nn.MSELoss()
num_epochs = 100

valid_accuracy = train_resac(model, device, optimizer, criterion, train_loader,valid_loader, num_epochs, scheduler=scheduler,tb=True)


#save model weights
save_path = "/usr/home/mwemaere/neuro/resac_mercator/Save/"

#torch.save(model.state_dict(), save_path+date+'model_conv1.pth')

#display loss
plot_valid_loss(valid_accuracy,save_path,date)





# Test
mean,std, l_im = test_resac(model,test_loader,device, get_im=[15,58,245])


print(mean)
print(std)


plot_test_ssh(l_im,save_path,date)