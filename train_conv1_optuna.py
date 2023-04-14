
# import modules
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture_conv1 import *
from plot_fct_conv1 import *
import optuna
from optuna.trial import TrialState


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

mean_ssh = torch.mean(ssh6)
mean_sst = torch.mean(sst6)
std_ssh = torch.std(ssh6)
std_sst = torch.std(sst6)

ssh3 -= mean_ssh
ssh6 -= mean_ssh
sst6 -= mean_sst
ssh3 /= std_ssh
ssh6 /= std_ssh
sst6 /= std_sst

ssh3 = torch.unsqueeze(ssh3,1)
ssh6 = torch.unsqueeze(ssh6,1)
sst6 = torch.unsqueeze(sst6,1)


def objective(trial):
    # prepare data
    batch_size = trial.suggest_int("batch_size",2,128,log=True)
    train_loader,valid_loader,test_loader = prepare_loaders(ssh3,ssh6,sst6,batch_size=batch_size)

    # create model
    model = RESAC_MERCATOR()





    # training 
    lr = trial.suggest_float("lr",5e-6,1e-2,log=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = RMSELoss(coeff=1)
    #criterion = torch.nn.MSELoss()
    num_epochs = trial.suggest_int("epoch",5,70)

    valid_accuracy = train_resac(model, device, optimizer, criterion, train_loader,valid_loader, num_epochs, verbose=0, scheduler=False, trial=trial)

    return valid_accuracy[-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=None)    

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

