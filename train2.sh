#!/bin/bash

echo 'Training models noaug  with CIFAR-100'
{
python main.py --dataset cifar100 --model mobilenetv1 --noaug
}
{
python main.py --dataset cifar100 --model vgg19 --noaug
echo 'Training Finished'
}
