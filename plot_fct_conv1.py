import matplotlib.pyplot as plt 
import numpy as np
import torch



def plot_valid_loss(valid_accuracy,save_path,date,save=True):
    plt.figure(1)
    plt.plot(valid_accuracy,label='ssh6')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(save_path+date+'valid_loss.png')
    plt.show()

def plot_test_ssh(l_im,save_path,date,cmap="coolwarm",save=True):
    # 4 col: ssh3, ssh6, ssh12 and ssh12 true
    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 
    
    for n,line in enumerate(l_im):
        [X,y,y_pred] = line
        min_val = min(torch.min(y),torch.min(y_pred),torch.min(X[0]))
        max_val = max(torch.max(y),torch.max(y_pred),torch.max(X[0]))
        im1 = axes[n,0].imshow(torch.squeeze(X[0]).cpu().numpy(),cmap=cmap,vmin=min_val, vmax=max_val)
        im2 = axes[n,1].imshow(torch.squeeze(y_pred).cpu().numpy(),cmap=cmap, vmin=min_val, vmax=max_val)
        im3 = axes[n,2].imshow(torch.squeeze(y).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)

        fig.colorbar(im1,ax=axes[n,0], orientation='vertical')
        fig.colorbar(im2,ax=axes[n,1], orientation='vertical')
        fig.colorbar(im3,ax=axes[n,2], orientation='vertical')
    
    cols=['ssh3','pred ssh6','true ssh6']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    
    if save:
        plt.savefig(save_path+date+'images.png')
    plt.show()

def plot_test_ssh_hist(l_im,save_path,date,cmap="plasma",save=True):
    # 4 col: ssh3, ssh6, ssh12 and ssh12 true
    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 
    
    for n,line in enumerate(l_im):
        [X,y,y_pred] = line
        min_val = min(torch.min(y),torch.min(y_pred),torch.min(X[0]))
        max_val = max(torch.max(y),torch.max(y_pred),torch.max(X[0]))
        histo, bin_edges = np.histogram(torch.squeeze(X[0]).cpu().numpy(), bins=1024, range=(0, 1))
        im1 = axes[n,0].plot(bin_edges[0:-1],histo)
        histo, bin_edges = np.histogram(torch.squeeze(y_pred).cpu().numpy(), bins=1024, range=(0, 1))
        im2 = axes[n,1].plot(bin_edges[0:-1],histo)
        histo, bin_edges = np.histogram(torch.squeeze(y).cpu().numpy(), bins=1024, range=(0, 1))
        im3 = axes[n,2].plot(bin_edges[0:-1],histo)


    
    cols=['ssh3','pred ssh6','true ssh6']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    
    if save:
        plt.savefig(save_path+date+'hists.png')
    plt.show()

