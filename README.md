# pytorch-classifications
Pytorch classification with Cifar-10, Cifar-100, and STL-10 using VGGs and ResNets

This repository is for classification using Pytorch toolkit.

## Setup

__Environment__: One or more multi-GPU with following libraries.
- [pytorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/)

## Arguments

__Training arguments__
- `--dataset`: choose datasets among cifar10, cifar100, and stl10 (default: cifar10)
- `--data_dir`: put the dataset root folder name (default: ~/dataset)
- `--batch_size`: batch size  (default: 128)
- `--lr`: learning rate (default: 0.1)
- `--lr_decay`: learning rate decay rate (default: 0.1)
- `--optimizer`: choose optimizer among sgd and adam (default: sgd)
- `--weight_decay`: weight decay (default: 0.0005)
- `--momentum`: momentum (default: 0.9)
- `--epochs`: the number of total training epochs (default: 300)
- `--save`: path to save the trained nets (default: trained_nets)
- `--save_epoch`: save every save_epochs (default: 10)
- `--ngpu`: the number of GPUs to use (default: 1)
- `--rand_seed`: seed for random num generator (default: 0)
- `--resume_model`: resume model from checkpoint (default: '')
- `--resume_opt`: resume optimizer from checkpoint (default: '')

__Model parameter arguments__
- `--model`, `-m`: model architecture (only chooseable among vgg9, 11, 16, 19, resnet18, 34, 50, 101, and 152) (default: vgg11)
- `--loss_name`: choose loss function among crossentropy and mse (default='crossentropy')

__Data parameter arguments__
- `--raw_data`: do not normalize data (default: False)
- `--noaug`: no data augmentation (default: False)
- `--label_corrupt_prob`: lable corrupt probability (default: 0.0)
- `--trainloader`: path to the dataloader with random labels (default: '')
- `--testloader`: path to the testloader with random labels (default: '')

__Other argument__
- `--idx`: the index for the repeated experiment (default: 0)

## To do
- Imagenet training will be added.
- Wide-ResNet and DenseNet can be easily implementable with including code
  - Little code changes are needed to do that (annotation code)
  
- This code is came from https://github.com/tomgoldstein/loss-landscape and modified.
