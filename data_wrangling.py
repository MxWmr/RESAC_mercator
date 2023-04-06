import xarray as xr 
import os
import numpy as np
from outils import generate_and_save
import torch

data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"


files = "glorys12v1_mod_product_001_030_*.nc"

filenames = os.path.join(data_path,files)

mf_ds = xr.open_mfdataset(filenames)

#print(mf_ds.head(10))

#print(mf_ds['time'].values[0],mf_ds['time'].values[-1])

#l_long = np.array(mf_ds['longitude'].values)    # 270
#l_lat = np.array(mf_ds['latitude'].values)  #210
l_t = np.array(mf_ds['time'].values) #9861

#sst_array = np.array(mf_ds['sst'].values) #9861x216x270
#np.save('SST_MERCATOR_full.npy',sst_array)


#ssh_array = np.array(mf_ds['sla'].values) #9861x216x270

# u_array = np.array(mf_ds['uo'].values)
# u_tensor = torch.from_numpy(u_array)
# print(u_tensor.shape)
# torch.save(u_tensor,'U_MERCATOR_1%12.pt')

# v_array = np.array(mf_ds['vo'].values)
# v_tensor = torch.from_numpy(v_array)
# print(v_tensor.shape)
# torch.save(v_tensor,'V_MERCATOR_1%12.pt')u_array = np.array(mf_ds['u0'].values)
# u_tensor = torch.from_numpy(u_array)#l_t = np.array(mf_ds['time'].values) #9861
# print(u_tensor.shape)
# torch.save(u_tensor,'U_MERCATOR_1%12.pt')


print(l_t[365*12+3])
print(l_t[365*13+3])