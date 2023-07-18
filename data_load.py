import torch
import math as mt
import numpy as np



class Dataset(torch.utils.data.Dataset):
    def __init__(self,l_files,n_files,path,file_name_sla,file_name_sst,file_name_u,file_name_v,batch_size=1,first_file=0):
        super().__init__()
        self.batch_size = batch_size
        self.file_name_sla = file_name_sla
        self.file_name_sst = file_name_sst
        self.file_name_u = file_name_u
        self.file_name_v = file_name_v
        self.l_files = l_files
        self.path = path
        self.n_files = n_files-first_file
        self.first_file = first_file
        self.pool = torch.nn.AvgPool2d(2,stride=(2,2))
        self.pool2 = torch.nn.AvgPool2d(4,stride=(4,4))

    def __len__(self):
        return (self.l_files//self.batch_size + 1)*self.n_files

    def __getitem__(self,i):
        len  = (self.l_files//self.batch_size + 1)*self.n_files
        
        i_f = i//(self.l_files//self.batch_size + 1)+self.first_file
        i_2 = i % (self.l_files//self.batch_size +1)

        if i >= len:
            raise IndexError()


        d_sla_12 = torch.load(self.path+self.file_name_sla+str(i_f)+'.pt')[:,:,:,:264]
        d_sst_12 = torch.load(self.path+self.file_name_sst+str(i_f)+'.pt')[:,:,:,:264]

        d_sla_6 = self.pool(torch.load(self.path+self.file_name_sla+str(i_f)+'.pt')[:,:,:,:264])
        d_sst_6 = self.pool( torch.load(self.path+self.file_name_sst+str(i_f)+'.pt')[:,:,:,:264])
        
        d_sla_3 = self.pool2( torch.load(self.path+self.file_name_sla+str(i_f)+'.pt')[:,:,:,:264])


        d_u = torch.load(self.path+self.file_name_u+str(i_f)+'.pt')[:,:,:,:264]
        d_v = torch.load(self.path+self.file_name_v+str(i_f)+'.pt')[:,:,:,:264]

        if self.batch_size*(i_2+1) <= self.l_files-2:

            sla_12 = d_sla_12[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            sst_12 = d_sst_12[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]

            sla_6 = d_sla_6[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            sst_6 = d_sst_6[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            
            sla_3 = d_sla_3[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]


            u = d_u[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            v = d_v[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            
        else:
            sla_12 = d_sla_12[i_2*self.batch_size+1:-1]
            sst_12 = d_sst_12[i_2*self.batch_size+1:-1]

            sla_6 = d_sla_6[i_2*self.batch_size+1:-1]
            sst_6 = d_sst_6[i_2*self.batch_size+1:-1]
            
            sla_3 = d_sla_3[i_2*self.batch_size+1:-1]


            u = d_u[i_2*self.batch_size+1:-1]
            v = d_v[i_2*self.batch_size+1:-1]

        return tuple([sla_3,sla_6,sla_12,sst_6,sst_12,u,v])    


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


