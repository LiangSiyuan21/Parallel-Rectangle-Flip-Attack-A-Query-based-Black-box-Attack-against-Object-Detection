import torch
import os
import json
import torch.nn as nn
from torchvision import models 
from surro_models import resnet_preact, resnet, pyramidnet, wrn, vgg, densenet
from surro_models.resnet import resnet152_denoise, resnet101_denoise
import numpy as np
import types

def load_torch_models(model_name):

    if model_name == "pyramidnet":
        TRAINED_MODEL_PATH = './results/pyramidnet_basic_110_84/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = pyramidnet.Network(json.load(fr)['model_config'])

            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'resnet_adv_4':
        TRAINED_MODEL_PATH = './results/resnet_adv_4/cifar-10_linf/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    elif model_name == 'resnet':
        TRAINED_MODEL_PATH = './results/resnet_basic_110/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'wrn':
        TRAINED_MODEL_PATH = './results/wrn_28_10/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = wrn.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'vgg':
        TRAINED_MODEL_PATH = './results/vgg_15_BN_64/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = vgg.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'dense':
        TRAINED_MODEL_PATH = './results/densenet_BC_100_12/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = densenet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])

    from advertorch.utils import NormalizeByChannelMeanStd

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    normalize = NormalizeByChannelMeanStd(
            mean=mean.tolist(), std=std.tolist())

    net = nn.Sequential(
        normalize,
        pretrained_model
    )

    net = net.cuda()
    net.eval()
    return net

def load_torch_models_tiny(model_name):
    if model_name == "pyramidnet":
        TRAINED_MODEL_PATH = './results_tiny/pyramidnet_basic_110_84/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = pyramidnet.Network(json.load(fr)['model_config'])
         
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'resnet_adv_4':
        TRAINED_MODEL_PATH = './results_tiny/resnet_adv_4/cifar-10_linf/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    elif model_name == 'resnet':
        TRAINED_MODEL_PATH = './results_tiny/resnet_basic_110/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'wrn':
        TRAINED_MODEL_PATH = './results_tiny/wrn_28_10/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = wrn.Network(json.load(fr)['model_config'])
            # print (torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'].keys())
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'vgg':
        TRAINED_MODEL_PATH = './results_tiny/vgg_15_BN_64/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = vgg.Network(json.load(fr)['model_config'])
            # print (torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'].keys())
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'dense':
        TRAINED_MODEL_PATH = './results_tiny/densenet_BC_100_12/00/'
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = densenet.Network(json.load(fr)['model_config'])
            # print (torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'].keys())
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])

    from advertorch.utils import NormalizeByChannelMeanStd

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    normalize = NormalizeByChannelMeanStd(
            mean=mean.tolist(), std=std.tolist())
    net = nn.Sequential(
        # Normalize(mean, std),
        normalize,
        pretrained_model
    )

    # if cuda:
    net = net.cuda()
    net.eval()
    return net


class Permute(nn.Module):

    def __init__(self, permutation = [2,1,0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        
        return input[:, self.permutation]


def load_torch_models_imagesub(model_name):
    if model_name == "VGG16":
        pretrained_model = models.vgg16_bn(pretrained=True)
    elif model_name == 'Resnet18':
        pretrained_model = models.resnet18(pretrained=True)
    elif model_name == 'Resnet34':
        pretrained_model = models.resnet34(pretrained=True)
    elif model_name == 'Resnet101':
        pretrained_model = models.resnet101(pretrained=True)
    # elif model_name == 'Squeezenet':
    #     pretrained_model = models.squeezenet1_1(pretrained=True)
    elif model_name == 'Googlenet':
        pretrained_model = models.googlenet(pretrained=True)
    elif model_name == 'Inception':
        pretrained_model = models.inception_v3(pretrained=True)
    elif model_name == 'Widenet':
        pretrained_model = models.wide_resnet50_2(pretrained=True)
    elif model_name == 'Adv_Denoise_Resnext101':
        pretrained_model = resnet101_denoise()
        loaded_state_dict = torch.load(os.path.join('./results/denoise/', model_name+".pytorch"))
        pretrained_model.load_state_dict(loaded_state_dict)
    # if 'defense' in state and state['defense']:
    #     net = nn.Sequential(
    #         Normalize(mean, std),
    #         Permute([2,1,0]),
    #         pretrained_model
    #     )
    # else:
    # net = nn.Sequential(
    #     Normalize(mean, std),
    #     pretrained_model
    # )

    from advertorch.utils import NormalizeByChannelMeanStd

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = NormalizeByChannelMeanStd(
            mean=mean.tolist(), std=std.tolist())

    if 'Denoise' in model_name:
        net = nn.Sequential(
            # Normalize(mean, std),
            normalize,
            Permute([2,1,0]),
            pretrained_model
        )
    
    else:

        net = nn.Sequential(
            # Normalize(mean, std),
            normalize,
            pretrained_model
        )
            

    # if cuda:
    net = net.cuda()
    net.eval()
    return net