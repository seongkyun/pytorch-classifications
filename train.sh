#!/bin/bash

echo 'Training VGGs/noaug with CIFAR-100'
{
python main.py --dataset cifar100 --model vgg11 --noaug
}
{
python main.py --dataset cifar100 --model vgg16 --noaug
echo 'Trainig is finished'
}

