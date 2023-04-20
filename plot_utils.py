import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import numpy as np
import torch

def plot_test_ssh(l_im,save_path,date,cmap="coolwarm",save=True):
    # 4 col: ssh3, ssh6, ssh12 and ssh12 true
    fig,axes=plt.subplots(nrows=len(l_im),ncols=4,figsize=(35,15)) 
    
    for n,line in enumerate(l_im):
        [ssh3,ssh6,ssh12,target] = line
        min_val = min(torch.min(ssh3),torch.min(ssh6),torch.min(ssh12),torch.min(target))
        max_val = max(torch.max(ssh3),torch.max(ssh6),torch.max(ssh12),torch.max(target))
        im1 = axes[n,0].imshow(torch.squeeze(ssh3).cpu().numpy(),cmap=cmap,vmin=min_val, vmax=max_val)
        im2 = axes[n,1].imshow(torch.squeeze(ssh6).cpu().numpy(),cmap=cmap, vmin=min_val, vmax=max_val)
        im3 = axes[n,2].imshow(torch.squeeze(ssh12).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im4 = axes[n,3].imshow(torch.squeeze(target).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)

        fig.colorbar(im1,ax=axes[n,0], orientation='vertical')
        fig.colorbar(im2,ax=axes[n,1], orientation='vertical')
        fig.colorbar(im3,ax=axes[n,2], orientation='vertical')
        fig.colorbar(im4,ax=axes[n,3], orientation='vertical')
    
    cols=['ssh3','pred ssh6','pred ssh12','true ssh12']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    
    if save:
        plt.savefig(save_path+date+'images.png')
    plt.show()




def plot_test_uv(l_im):


    fig,axes=plt.subplots(nrows=len(l_im),ncols=2,figsize=(35,15)) 

    clim=(100,-100)

    l_x = l_im[0][1][2].shape[2]
    l_y = l_im[0][1][2].shape[3]
    for n,line in enumerate(l_im):
        [X,y,y_pred] = line
        long,lat = np.meshgrid(np.linspace(-100,100,l_x),np.linspace(-100,100,l_y))
        im = axes[n,0].quiver(torch.squeeze(y[2]).cpu().numpy(),torch.squeeze(y[3]).cpu().numpy(),color='g')
        im = axes[n,1].quiver(torch.squeeze(y_pred[2]).cpu().numpy(),torch.squeeze(y_pred[3]).cpu().numpy(),color='g')

    
    axes[0,0].set_title('true currents')
    axes[0,1].set_title('pred currents')

    fig.tight_layout()
    plt.show()
