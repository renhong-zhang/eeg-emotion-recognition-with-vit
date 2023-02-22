from __future__ import print_function
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import optuna
from tqdm import tqdm
from vit_pytorch import ViT
from torch.utils.data import DataLoader, Dataset
import pickle as pickle
import numpy as np
import random

epochs = 100
gamma = 0.9   
seed = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1024
train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': batch_size}

cuda_kwargs = {'num_workers': 8,
               'pin_memory': True,
               'shuffle': True}
    
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = Truce  #*add: train more fastly
    else:
        torch.backends.cudnn.deterministic = False

wt_name = 'cwt'
traindatafile='/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_03/mydata/final_data_train_'+wt_name+'.pkl'
trainlabelfile='/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_03/mydata/final_lables_train_'+wt_name+'.pkl'
validdatafile='/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_03/mydata/final_data_test_'+wt_name+'.pkl'
validlabelfile='/home/renhong_zhang/path-to-project/DEAP/test_out/my_erVITer_03/mydata/final_lables_test_'+wt_name+'.pkl'

class eegDataset(Dataset):
    def __init__(self, datafile,labelfile,valence_arouse = 0):
     
        self.valence_arouse = valence_arouse
        with open(datafile,'rb') as filepath:
            self.new_dataset = pickle.load(filepath)
            filepath.close()
            self.new_dataset = np.array(self.new_dataset)
        with open(labelfile,'rb') as filepath:
            self.new_labels = pickle.load(filepath)
            filepath.close()
            self.new_labels = np.array(self.new_labels)
        _,chans,channels,rows,cols=self.new_dataset.shape
        self.chans = chans
        self.new_dataset = self.new_dataset.reshape(-1,channels,rows,cols)
            
        """
        nums,freqs,chans,wavesegs=self.new_dataset.shape
        if chans>wavesegs:
            self.new_dataset.resize((nums,freqs,chans,chans),refcheck=False)
        else:
            if chans<wanvesegs:
                self.new_dataset.resize((nums,freqs,wavesegs,wavesegs),refcheck=False)
        """
        
    def __len__(self):
        return len(self.new_dataset)

    def __getitem__(self, index):

        image_array = self.new_dataset[index]
        label = self.new_labels[int(index/self.chans),self.valence_arouse]
        if label >= 5:
          label = 1
        else:
          label = 0
         
        return image_array, label

train_data = eegDataset(traindatafile,trainlabelfile,0)
valid_data = eegDataset(validdatafile, validlabelfile,0)

train_loader = DataLoader(train_data,**train_kwargs)
test_loader = DataLoader(valid_data, **test_kwargs)

def train(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in tqdm(train_loader):
        
        data, target = data.to(torch.float32).to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Train Epoch: {} Loss: {:.6f}'.format(
        epoch, loss.item()))
    
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            
            data, target = data.to(torch.float32).to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  #* sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # * get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        val_acc))
    
    return val_acc

def Objective(trial):
    
    dim = trial.suggest_categorical('dim',[32, 64, 128])
    patch_size = 16
    depth = trial.suggest_categorical('depth',[8, 16, 32])
    heads = trial.suggest_categorical('heads',[8, 16, 32])
    mlp_dim = trial.suggest_categorical('mlp_dim',[128, 512, 1024])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop","SGD"])
    lr = trial.suggest_float("lr", 3e-5, 1e-1, log=True)
    print('dim:', dim, 'mlp_dim:',mlp_dim, 'depth:',depth, 'heads:',heads)
    model = ViT(
        dim=dim,
        image_size=48,
        patch_size=patch_size,
        num_classes=2,
        depth=depth, # *number of transformer blocks
        heads=heads, # *number of multi-channel attention
        mlp_dim=mlp_dim,
        channels=1,
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # *optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # *scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, criterion, device, train_loader, optimizer, epoch)
        val_acc = test(model, device, test_loader)
        scheduler.step()
        
        if 0:
            torch.save(model.state_dict(), "test_transformer.pt")
    
        trial.report(val_acc, epoch)

        # *Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(Objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

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
