import os
import torch, torchvision
import models.vgg as vgg
import models.resnet as resnet
import models.densenet as densenet
#import models_stl.vgg as vgg_stl
#import models_stl.resnet as resnet_stl
#import models_stl.densenet as densenet_stl

# map between model name and function
#print(args.dataset)
import sys
#sys.exit()
global num_class
global d_name
models = {
    'vgg9'                  : vgg.VGG9,
    'vgg11'                 : vgg.VGG11,
    'vgg16'                 : vgg.VGG16,
    'vgg19'                 : vgg.VGG19,
    #'densenet121'           : densenet.DenseNet121,
    'resnet18'              : resnet.ResNet18,
    #'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34'              : resnet.ResNet34,
    #'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50'              : resnet.ResNet50,
    #'resnet50_noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    #'resnet101_noshort'     : resnet.ResNet101_noshort,
    'resnet152'             : resnet.ResNet152,
    #'resnet152_noshort'     : resnet.ResNet152_noshort,
    #'resnet20'              : resnet.ResNet20,
    #'resnet20_noshort'      : resnet.ResNet20_noshort,
    #'resnet32_noshort'      : resnet.ResNet32_noshort,
    #'resnet44_noshort'      : resnet.ResNet44_noshort,
    #'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
    #'resnet56'              : resnet.ResNet56,
    #'resnet56_noshort'      : resnet.ResNet56_noshort,
    #'resnet110'             : resnet.ResNet110,
    #'resnet110_noshort'     : resnet.ResNet110_noshort,
    #'wrn56_2'               : resnet.WRN56_2,
    #'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    #'wrn56_4'               : resnet.WRN56_4,
    #'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    #'wrn56_8'               : resnet.WRN56_8,
    #'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    #'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    #'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
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

    net.eval()
    return net