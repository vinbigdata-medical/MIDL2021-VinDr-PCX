import torch.nn as nn
import torch
import os
import numpy as np
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet121, densenet161, densenet169, densenet201, resnet18, resnet34, resnet50, resnet101
from model.models import Ensemble

def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))

def get_models(cfg):
    if isinstance(cfg.backbone, str):
        return get_model(cfg)
    elif isinstance(cfg.backbone, list):
        backbones = cfg.backbone
        ids = cfg.id
        ckp_paths = cfg.ckp_path
        models = Ensemble()
        for i in range(len(backbones)):
            cfg.backbone = backbones[i]
            cfg.id = ids[i]
            cfg.ckp_path = ckp_paths[i]
            model= get_model(cfg)
            if os.path.isfile(cfg.ckp_path):
                load_ckp(model, cfg.ckp_path, torch.device("cpu"), cfg.parallel, strict=True)
            models.append(model)
        cfg.backbone = backbones
        cfg.id = ids
        cfg.ckp_path = ckp_paths
        return models

def get_model(cfg):
    if cfg.backbone == 'resnest':
        model = get_resnest(cfg.id, cfg.num_classes, cfg.pretrained)
        
    elif cfg.backbone == 'efficient' or cfg.backbone == 'efficientnet':
        model = get_efficientnet(cfg.id, cfg.num_classes, cfg.pretrained)

    elif cfg.backbone == 'dense' or cfg.backbone == 'densenet':
        model = get_densenet(cfg.id, cfg.num_classes, cfg.pretrained)
        
    elif cfg.backbone == 'resnet':
        model = get_resnet(cfg.id, cfg.num_classes, cfg.pretrained)

    else:
        raise Exception("Not support this model!!!!")

    return model

def get_resnest(id_model, num_classes, pretrained=True):
    if id_model == '50':
        pre_name = resnest50
    elif id_model == '101':
        pre_name = resnest101
    elif id_model == '200':
        pre_name = resnest200
    else:
        pre_name = resnest269
    model = pre_name(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features,
                        out_features=len(num_classes), bias=True)
    return model

def get_densenet(id_model, num_classes, pretrained=True):
    if id_model == '121':
        pre_name = densenet121
    elif id_model == '161':
        pre_name = densenet161
    elif id_model == '169':
        pre_name = densenet169
    else:
        pre_name = densenet201
    model = pre_name(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features=num_features,
                        out_features=len(num_classes), bias=True)
    return model

def get_resnet(id_model, num_classes, pretrained=True):
    if id_model == '34':
        pre_name = resnet34
    elif id_model == '18':
        pre_name = resnet18
    elif id_model == '50':
        pre_name = resnet50
    else:
        pre_name = resnet101
    model = pre_name(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features,
                        out_features=len(num_classes), bias=True)
    return model

def get_efficientnet(id_model, num_classes, pretrained=True):
    pre_name = 'efficientnet-'+id_model
    if pretrained:
        model = EfficientNet.from_pretrained(pre_name)
    else:
        model = EfficientNet.from_name(pre_name)
    for param in model.parameters():
        param.requires_grad = True        
    num_features = model._fc.in_features
    model._fc = nn.Linear(in_features=num_features,
                            out_features=len(num_classes), bias=True)
    return model

def load_ckp(model, ckp_path, device, parallel=False, strict=True):
    """Load checkpoint

    Args:
        ckp_path (str): path to checkpoint

    Returns:
        int, int: current epoch, current iteration
    """
    ckp = torch.load(ckp_path, map_location=device)
    if parallel:
        model.module.load_state_dict(
            ckp['state_dict'], strict=strict)
    else:
        model.load_state_dict(ckp['state_dict'], strict=strict)

    return ckp['epoch'], ckp['iter']

def get_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    device = str(device)
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    
    return torch.device('cuda:'+device if cuda else 'cpu')

def get_metrics(preds, labels, metrics_dict, thresh_val=0.5):

    running_metrics = dict.fromkeys(metrics_dict.keys(), 0.0)
    for key in list(metrics_dict.keys()):
        if key in ['f1', 'precision', 'recall', 'specificity', 'sensitivity', 'acc']:
            running_metrics[key] = tensor2numpy(metrics_dict[key](
                preds, labels, thresh_val))
        else:
            running_metrics[key] = tensor2numpy(metrics_dict[key](
                preds, labels))

    return running_metrics

def get_str(metrics, mode, s):
    for key in list(metrics.keys()):
        if key == 'loss':
            s += "{}_{} {:.3f} - ".format(mode, key, metrics[key])
        else:
            metric_str = ' '.join(
                map(lambda x: '{:.5f}'.format(x), metrics[key]))
            s += "{}_{} {} - ".format(mode, key, metric_str)
    s = s[:-2] + '\n'
    return s

def tensor2numpy(input_tensor):
    # device cuda Tensor to host numpy
    return input_tensor.cpu().detach().numpy()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def lrfn(epoch):
    LR_START = 0.00001
    LR_MAX = 0.0004
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 10
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr