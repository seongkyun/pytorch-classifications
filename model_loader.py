import os
import torch, torchvision
import models.alexnet as alexnet
import models.vgg as vgg
import models.resnet as resnet
import models.densenet as densenet
import models.mobilenetv1 as mobilenetv1

# map between model name and function
import sys
global num_class
global d_name
models = {
    'alexnet'               : alexnet.AlexNet,
    'vgg9'                  : vgg.VGG9,
    'vgg11'                 : vgg.VGG11,
    'vgg16'                 : vgg.VGG16,
    'vgg19'                 : vgg.VGG19,
    'resnet18'              : resnet.ResNet18,
    'resnet34'              : resnet.ResNet34,
    'resnet50'              : resnet.ResNet50,
    'resnet101'             : resnet.ResNet101,
    'resnet152'             : resnet.ResNet152,
    'mobilenetv1'           : mobilenetv1.MobileNet
    
}

def get_args(args):
    global num_class
    global d_name
    if args.dataset == 'cifar10':
        num_class = 10
        d_name = 'c10'
    elif args.dataset == 'stl10':
        num_class = 10
        d_name = 's'
    elif args.dataset == 'cifar100':
        num_class = 100
        d_name = 'c100'
    elif args.dataset == 'imagenet':
        num_class = 1000
        d_name = 'i'
    else:
        print('ERROR::WRONG DATASET')
        print('Choose among cifar10, cifar100, stl10, and imagenet')
        sys.exit()
    return None

def load(model_name, model_file=None, data_parallel=False):
    global num_class
    global d_name

    if d_name == 's':
        net = models[model_name](input_size=96, num_class=num_class)
    elif d_name == 'i':
        print('imagenet is not supported yet.')
    else:
        net = models[model_name](input_size=32, num_class=num_class)

    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    #net.eval()
    return net