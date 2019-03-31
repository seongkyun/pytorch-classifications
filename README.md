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

## Available networks
- __!IMPORTANT!__ : All the architecture of networks are modified
  - The fully-connected layers are added for my experiment(to output 128-dimension vectors).
  - If you want to ues original models, then just remove one fully-connected layer among 2 of each architecthre.

- VGGs
  - VGG19: `--model vgg9`
  - VGG11: `--model vgg11`
  - VGG16: `--model vgg16`
  - VGG19: `--model vgg19`
- ResNets
  - ResNet18: `--model resnet18`
  - ResNet34: `--model resnet34`
  - ResNet50: `--model resnet50`
  - ResNet101: `--model resnet101`
  - ResNet152: `--model resnet152`
- MobileNets
  - MobileNet V1: `--model mobilenetv1`
  - MobileNet V2: Not yet supported (will be added)
- DenseNets
  - DenseNet121: Not yet supported (will be added)
  - DenseNet169: Not yet supported (will be added)
  - DenseNet201: Not yet supported (will be added)
  - DenseNet161: Not yet supported (will be added)

## Available datasets
- CIFAR-10: `--dataset cifar10`
- CIFAR-100: `--dataset cifar100`
- STL-10: `--dataset stl10`
- IMAGENET(ILSVRC12): Not yet supported (will be added)

## Experimental results
- Every hyper parameters are same with default settings
- Cifar-10 dataset
  - VGG11: 91.94 %
  - VGG16: 92.98 %
  - VGG19: 93.24 %
  - ResNet50: 94.06 %
- Cifar-100 dataset
  - VGG11: 70.53 %
  - VGG16: 74.03 %
  - VGG19: 72.66 %
  - ResNet50: 79.57 %
- STL-10 dataset
  - VGG11: 79.62 %
  - VGG16: 80.20 %
  - VGG19: 70.17 %
  - ResNet34: 87.23 %

## To do
- Imagenet training will be added.
- Wide-ResNet and DenseNet can be easily implementable with including code
  - Little code changes are needed to do that (annotation code)
  
- This code is came from https://github.com/tomgoldstein/loss-landscape and modified.
