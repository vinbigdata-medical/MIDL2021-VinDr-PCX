from easydict import EasyDict as edict
import json, os
from model.classifier import Pediatric_Classifier
from utils.metrics import F1, AUC, Recall, Specificity
from torch.nn import BCELoss, BCEWithLogitsLoss
import warnings
import pandas as pd
import numpy as np
from data.dataset import create_loader
import torch
import argparse

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--config', type=str, default='config/test_config.json', help='*.config path')

    opt = parser.parse_args()
    print(opt)

    with open(opt.config) as f:
        cfg = edict(json.load(f))

    loss_func = BCEWithLogitsLoss()

    torch.cuda.set_device(cfg.device)
    val_loader = create_loader(cfg.dev_csv, cfg, mode='val')
    test_loader = create_loader(cfg.test_csv, cfg, mode='test')

    metrics_dict = {'auc':AUC(), 'sensitivity':Recall(), 'specificity':Specificity(), 'f1':F1()}

    #------------------------------- additional config for ensemble ---------------------------------------
    # model_names=[
    #     'dense',
    #     'resnet',
    #     'dense',
    #     # 'efficient',
    #     #'resnest'
    #     ]
    # ids = [
    #     '121',
    #     '101',
    #     '169',
    #     # 'b4',
    #     #'101'
    #     ]

    # # ckp_paths = [
    # #     'experiment/DenseNet121_data2203_finetune_chexpmic_cutmix/checkpoint/best.ckpt',
    # #     'experiment/Resnet101_data2203_finetune_chexpmic_cutmix/checkpoint/best.ckpt',
    # #     'experiment/DenseNet169_data2203_finetune_chexpmic_cutmix/checkpoint/best.ckpt',
    # # ]

    # # ckp_paths = [
    # #     'experiment/DenseNet121_data2203_finetune_chexpmic_mixup/checkpoint/best.ckpt',
    # #     'experiment/Resnet101_data2203_finetune_chexpmic_mixup/checkpoint/best.ckpt',
    # #     'experiment/DenseNet169_data2203_finetune_chexpmic_mixup/checkpoint/best.ckpt'
    # # ]

    # ckp_paths = [
    #     'experiment/DenseNet121_data2203_finetune_chexpmic/checkpoint/best.ckpt',
    #     'experiment/Resnet101_data2203_finetune_chexpmic/checkpoint/best.ckpt',
    #     'experiment/DenseNet169_data2203_finetune_chexpmic/checkpoint/best.ckpt',
    #     ]

    # cfg.backbone = model_names
    # cfg.id = ids
    # cfg.ckp_path = ckp_paths
    #------------------------------------------------------------------------------------------------------

    n_boostrap = 10000

    pediatric_classifier = Pediatric_Classifier(cfg, loss_func, metrics_dict)
    if not isinstance(cfg.ckp_path, list):
        pediatric_classifier.load_ckp(cfg.ckp_path)
    pediatric_classifier.thresholding(val_loader)

    metrics, ci_dict = pediatric_classifier.test(val_loader, get_ci=True, n_boostrap=n_boostrap)
    for key in metrics_dict.keys():
        if key != 'loss':
            print(key, metrics[key], metrics[key].mean())
            metrics[key] = np.append(metrics[key],metrics[key].mean())
            metrics[key] = list(map(lambda a: round(a, 3), metrics[key]))
            ci_dict[key] = list(map(lambda a: round(a, 3), ci_dict[key]))
            metrics[key][-1] = str(metrics[key][-1])+'('+str(ci_dict[key][0])+'-'+str(ci_dict[key][1])+')'
    metrics.pop('loss')
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv('val_result.csv', index=False) 

    metrics, ci_dict = pediatric_classifier.test(test_loader, get_ci=True, n_boostrap=n_boostrap)
    for key in metrics_dict.keys():
        if key != 'loss':
            print(key, metrics[key], metrics[key].mean())
            metrics[key] = np.append(metrics[key],metrics[key].mean())
            metrics[key] = list(map(lambda a: round(a, 3), metrics[key]))
            ci_dict[key] = list(map(lambda a: round(a, 3), ci_dict[key]))
            metrics[key][-1] = str(metrics[key][-1])+'('+str(ci_dict[key][0])+'-'+str(ci_dict[key][1])+')'
    metrics.pop('loss')
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv('test_result.csv', index=False,)  