# Full pipeline to train a Point Transformer Model on ModelNet40
# Source : NPM3D TP6 - PointNet

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import madgrad

from point_transformer_classif import PointTransformerClassif
from point_transformer_block import PointTransformerBlock
from transition_down import TransitionDown
from dataset import ModelNetDataLoader


def train(model, device, optimizer, scheduler, train_loader, test_loader=None, epochs=100, val_step=5):
   
    val_accs = []
    train_accs = []
    best_val_acc = -1.0
    loss=0

    for epoch in tqdm(range(epochs), position=0, leave=True): 
        model.train()
        correct = total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100. * correct / total
        train_accs.append(train_acc)

        if (epoch+1) % val_step  == 0:
            model.eval()
            correct = total = 0
            if test_loader:
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_acc = 100. * correct / total
                val_accs.append(val_acc)
                print('\n Epoch: %d, Train accuracy: %.1f %%, Test accuracy: %.1f %%' %(epoch+1, train_acc, val_acc))
            if val_accs[-1] > best_val_acc:
                torch.save(model.state_dict(), 'checkpoint.pth')
        else:
            print('\n Epoch: %d, Train accuracy: %.1f %%' %(epoch+1, train_acc))

        scheduler.step()

    return train_accs, val_accs

if __name__ == '__main__':

    lr = 1e-3
    wd = 1e-4
    opt = 'MadGrad' # 'Adam', 'SGD', 'MadGrad'
    batch_size = 64
    epochs = 100

    # device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        ModelNetDataLoader('data/modelnet40_normal_resampled/', split='train', process_data=True, transforms=None), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        ModelNetDataLoader('data/modelnet40_normal_resampled/', split='test', process_data=True, transforms=None), 
        batch_size=64, 
        shuffle=True
    )

    # model
    print('Creating Model...')
    model = PointTransformerClassif().to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=wd
        )
    elif opt == 'MadGrad':
        optimizer = madgrad.MADGRAD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs*6//10,epochs*8//10], gamma=0.1)

    print('Training...')
    train_accs, val_accs = train(model, device, optimizer, scheduler, train_loader, test_loader, val_step=10, epochs=epochs)
    plt.plot(train_accs)