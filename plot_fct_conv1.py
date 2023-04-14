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

    clim=(100,-100)

    for n,line in enumerate(l_im):
        [X,y,y_pred] = line
        im = axes[n,0].imshow(torch.squeeze(X[0]).cpu().numpy(),cmap=cmap)
        im = axes[n,1].imshow(torch.squeeze(y_pred).cpu().numpy(),cmap=cmap)
        im = axes[n,2].imshow(torch.squeeze(y).cpu().numpy(),cmap=cmap)
    
    cols=['ssh3','pred ssh6','true ssh6']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()

    if save:
        plt.savefig(save_path+date+'ims.png')
    plt.show()


