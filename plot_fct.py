import matplotlib.pyplot as plt 
import numpy as np
import torch


def plot_train_loss(train_accuracy,save_path,date):
    plt.figure(1)
    plt.plot(train_accuracy,label=['ssh6','ssh12','u','v'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path+date+'train_loss.png')
    plt.show()

def plot_valid_loss(valid_accuracy,save_path,date):
    plt.figure(1)
    plt.plot(valid_accuracy,label=['ssh6','ssh12','u','v'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path+date+'valid_loss.png')
    plt.show()

def plot_test_ssh(l_im,cmap="coolwarm"):
    # 4 col: ssh3, ssh6, ssh12 and ssh12 true
    fig,axes=plt.subplots(nrows=len(l_im),ncols=4,figsize=(35,15)) 

    clim=(100,-100)

    for n,line in enumerate(l_im):
        [X,y,y_pred] = line
        im = axes[n,0].imshow(torch.squeeze(X[0]).cpu().numpy(),cmap=cmap)
        im = axes[n,1].imshow(torch.squeeze(y_pred[0]).cpu().numpy(),cmap=cmap)
        im = axes[n,2].imshow(torch.squeeze(y_pred[1]).cpu().numpy(),cmap=cmap)
        im = axes[n,3].imshow(torch.squeeze(y[1]).cpu().numpy(),cmap=cmap)
    
    cols=['ssh3','pred ssh6','pred ssh12','true ssh12']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
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


def plot_uv(u,v):
    plt.figure(1)
    plt.quiver(torch.squeeze(u).numpy(),torch.squeeze(v).numpy())
    plt.show()


def plot_degrad(ssh3,ssh6,ssh12):
    fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(35,15)) 
    axes[0].imshow(torch.squeeze(ssh3).cpu().numpy())
    axes[1].imshow(torch.squeeze(ssh6).cpu().numpy())
    axes[2].imshow(torch.squeeze(ssh12).cpu().numpy())
    axes[0].set_title('ssh3')
    axes[1].set_title('ssh6')
    axes[2].set_title('ssh12')

    fig.tight_layout()
    plt.show()