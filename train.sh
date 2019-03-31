#!/bin/bash

echo 'Training MobileNet V1 with CIFAR-100'
{
python main.py --dataset cifar100 --model mobilenetv1
echo 'Training Finished'
}

echo 'Training MobileNet V1 with STL-10'
{
python main.py --dataset stl10 --model mobilenetv1
echo 'Training Finished'
}