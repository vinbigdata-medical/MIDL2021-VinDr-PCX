import torch.nn as nn
import torch
import os

def save_dense_backbone(model, ckp_path):
    if os.path.exists(os.path.dirname(ckp_path)):
        torch.save({'state_dict': model.features.state_dict(),
                    'epoch': 0, 'iter': 0}, ckp_path)
    else:
        print("Save path not exist!!!")

def load_dense_backbone(model, ckp_path, device, strict):
    if os.path.exists(os.path.dirname(ckp_path)):
        ckp = torch.load(ckp_path, map_location=device)
        model.features.load_state_dict(ckp['state_dict'], strict=strict)
    else:
        print("Save path not exist!!!")

def save_resnet_backbone(model, ckp_path):
    if os.path.exists(os.path.dirname(ckp_path)):
        save_model = model
        delattr(save_model,'avgpool')
        delattr(save_model,'fc')
        torch.save({'state_dict': save_model.state_dict(),
                    'epoch': 0, 'iter': 0}, ckp_path)
    else:
        print("Save path not exist!!!")

def load_resnet_backbone(model, ckp_path, device, strict):
    if os.path.exists(os.path.dirname(ckp_path)):
        ckp = torch.load(ckp_path, map_location=device)
        model.load_state_dict(ckp['state_dict'], strict=strict)
    else:
        print("Save path not exist!!!")

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def freeze(self):
        for module in self:
            for param in module.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        for module in self:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x):
        y = []
        for module in self:
            y.append(nn.Sigmoid()(module(x)))
        y = torch.stack(y, -1)

        return y

class Stacking(nn.Module):
    def __init__(self, n_models=3):
        super(Stacking, self).__init__()
        self.ensemble = nn.Linear(in_features=n_models, out_features=1, bias=False)
        torch.nn.init.xavier_uniform(self.ensemble.weight)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        y = self.ensemble(x).squeeze(-1)
        return y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count