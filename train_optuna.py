
# import modules
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture import *
from plot_fct import *
import optuna
from optuna.trial import TrialState


if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')


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




def objective(trial):


    # prepare data
    batch_size = trial.suggest_int("batch_size",6,128,log=True)
    train_loader,valid_loader,test_loader = prepare_loaders(ssh3,ssh6,ssh12,sst6,sst12,u12,v12,batch_size=batch_size)

    # create model
    model = RESAC_MERCATOR()



    # training 
    lr = trial.suggest_float("lr",1e-7,1e-2,log=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, custom_scheduler)
    criterion = RMSELoss()
    num_epochs = trial.suggest_int("epoch",20,80)


    train_accuracy, valid_accuracy = train_resac(model, device, optimizer, criterion, train_loader,valid_loader, num_epochs, verbose=0, scheduler=False, trial=trial)


    # save model weights
    # save_path = "/usr/home/mwemaere/neuro/resac_mercator/Save/"
    # date = datetime.now().strftime("%m_%d_%H:%M_")
    # torch.save(model.state_dict(), save_path+date+'model.pth')

    # display loss
    # plot_train_loss(train_accuracy,save_path,date)
    # plot_valid_loss(valid_accuracy,save_path,date)


    return valid_accuracy[-1][1]




if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=600)    

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


