import numpy as np
import torch



def downsample_12to3(im):
    t_im = torch.from_numpy(im)
    pool = torch.nn.AvgPool2d(4,stride=(4,4))

    return pool(t_im)

def downsample_12to6(im):
    t_im = torch.from_numpy(im)
    pool = torch.nn.AvgPool2d(2,stride=(2,2))

    return pool(t_im)

# sst_array = np.load('SST_MERCATOR_full.npy')
# print(sst_array.shape)
# torch.save(sst_array,'SST_MERCATOR_1%12.pt')

# sst_array3 = downsample_12to3(sst_array)
# print(sst_array3.shape)
# torch.save(sst_array3,'SST_MERCATOR_1%3.pt')

# sst_array6 = downsample_12to6(sst_array)
# print(sst_array6.shape)
# torch.save(sst_array6,'SST_MERCATOR_1%6.pt')

ssh_array = np.load('SSH_MERCATOR_full.npy')
t_ssh_array = torch.from_numpy(ssh_array)
print(t_ssh_array.shape)
torch.save(t_ssh_array,'SSH_MERCATOR_1%12.pt')

ssh_array3 = downsample_12to3(ssh_array)
print(ssh_array3.shape)
torch.save(ssh_array3,'SSH_MERCATOR_1%3.pt')

ssh_array6 = downsample_12to6(ssh_array)
print(ssh_array6.shape)
torch.save(ssh_array6,'SSH_MERCATOR_1%6.pt')

sst_array = np.load('SST_MERCATOR_full.npy')
t_sst_array = torch.from_numpy(sst_array)
print(t_sst_array.shape)
torch.save(t_sst_array,'SST_MERCATOR_1%12.pt')

