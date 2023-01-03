"""
denoising MNIST using AUtoencoders using MNIST
"""

# region packages

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from utils_mnist import *
from mnist_models import *
from mypyutils.utils_dl import *
from mypyutils.utils_general import *
from pytorch_model_summary import summary
# endregion


device = torch.device('mps')

# -------------------------
# directories
# -------------------------
path_mnist = '/Users/mina/Documents/Academics/Data/Benchmarks'


# -------------------------
# load data
# -------------------------
batch_size = 128
img_chan1 = 1
img_dim = 28

transforms_train = transforms.Compose([MaxNormalize(255)])
transforms_test = transforms.Compose([MaxNormalize(255), AddNoise(0.5), Clip(0, 1)])

dataloader_train, dataloader_val, dataloader_test = mnist_dataloaders(path_mnist, transforms_train, transforms_test,
                                                                      batch_size, vectorize=False)

# -------------------------
# model training
# -------------------------
n_epochs = 20
lr_init = 1e-3

model = MyConvAutoEncoder1(img_chan1)
# model = MyDenseAutoEncoder1(img_dim ** 2)
# print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=False))

model.to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-4, amsgrad=False)

training_performance, model_trained, best_model = train_autoencoder(model, loss_func, optimizer, n_epochs, lr_init,
                                                                    device, dataloader_train, dataloader_val,
                                                                    min_epoch_num=0, lr_schedule1=True, n_epoch_val=1,
                                                                    early_stopping=False, patience=None,
                                                                    stop_criterion='loss', save_path=None)

_, (t1, _) = next(enumerate(dataloader_test))
output = model_trained(t1.to(device))
plt.figure()
plt.subplot(121)
plt.imshow(t1[10, 0], cmap='gray')
plt.subplot(122)
plt.imshow(output[10, 0].detach().cpu().numpy(), cmap='gray')

# plt.imshow(output[10].detach().cpu().numpy().reshape((28, 28)), cmap='gray')


loss_all = training_performance['loss_all_minibatches']