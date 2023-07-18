import matplotlib.pyplot as plt 
#plt.style.use('dark_background')
import numpy as np
import torch

def plot_test_ssh(l_im,save_path,date,cmap="coolwarm",save=True):
    # 4 col: ssh3, ssh6, ssh12 and ssh12 true
    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 
    
    for n,line in enumerate(l_im):
        [ssh6,ssh12,_,_,target,_,_] = line
        min_val = min(torch.min(ssh6),torch.min(ssh12),torch.min(target))
        max_val = max(torch.max(ssh6),torch.max(ssh12),torch.max(target))
        im2 = axes[0].imshow(torch.squeeze(ssh6).cpu().numpy(),cmap=cmap, vmin=min_val, vmax=max_val)
        im3 = axes[1].imshow(torch.squeeze(ssh12).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im4 = axes[2].imshow(torch.squeeze(target).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)

        fig.colorbar(im2,ax=axes[0], orientation='horizontal',label='meter')
        fig.colorbar(im3,ax=axes[1], orientation='horizontal',label='meter')
        fig.colorbar(im4,ax=axes[2], orientation='horizontal',label='meter')
    
    cols=['pred ssh6','pred ssh12','true ssh12']
    for ax, col in zip(axes, cols):
        ax.set_title(col)

    fig.tight_layout()
    
    if save:
        plt.savefig(save_path+date+'images.png')
    plt.show()


def plot_test_uv(l_im,save_path,date,cmap="inferno",save=True):

    fig,axes=plt.subplots(nrows=len(l_im),ncols=4,figsize=(35,15)) 
    
    for n,line in enumerate(l_im):
        [s2,s3,u_pred,v_pred,s4,u,v] = line
        min_val = min(torch.min(u),torch.min(v),torch.min(u_pred),torch.min(v_pred))
        max_val = max(torch.max(u),torch.max(v),torch.max(u_pred),torch.max(v_pred))
        im1 = axes[0].imshow(torch.squeeze(u_pred).cpu().numpy(),cmap=cmap)
        im2 = axes[1].imshow(torch.squeeze(v_pred).cpu().numpy(),cmap=cmap)
        im3 = axes[2].imshow(torch.squeeze(u).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im4 = axes[3].imshow(torch.squeeze(v).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)


    cols=['u_pred','v_pred','u_pred','v_pred']
    for ax, col in zip(axes, cols):
        ax.set_title(col)

    fig.tight_layout()
    
    if save:
        plt.savefig(save_path+date+'uv.png')
    plt.show()


def plot_test_uv2(l_im,save_path,date,save=True):


    fig,axes=plt.subplots(nrows=len(l_im),ncols=2,figsize=(35,15)) 

    clim=(100,-100)

    l_x = l_im[0][3].shape[2]
    l_y = l_im[0][3].shape[3]
    for n,line in enumerate(l_im):
        [_,_,u_pred,v_pred,_,u,v] = line
        long,lat = np.meshgrid(np.linspace(-100,100,l_x),np.linspace(-100,100,l_y))
        im = axes[0].quiver(torch.squeeze(u_pred).cpu().numpy(),torch.squeeze(v_pred).cpu().numpy(),color='g')
        im = axes[1].quiver(torch.squeeze(u).cpu().numpy(),torch.squeeze(v).cpu().numpy(),color='g')

    
    axes[0].set_title('pred currents')
    axes[1].set_title('true currents')

    fig.tight_layout()
    if save:
        plt.savefig(save_path+date+'uv2.png')
    plt.show()
