import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# from data.dataset import ImageDataset_full
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

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

cfg_path = './config/test_config.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

loss_func = BCEWithLogitsLoss()

# data_dir = '/home/tungthanhlee/bdi_xray/data/images'
data_dir = '/home/single1/BACKUP/thanhtt/assigned_jpeg'

torch.cuda.set_device(cfg.device)
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val')
test_loader = create_loader(cfg.test_csv, data_dir, cfg, mode='test')

metrics_dict = {'auc':AUC(), 'sensitivity':Recall(), 'specificity':Specificity(), 'f1':F1()}

id_leaf = [2,4,5,6,7,8]
id_obs = [0,1,2,3,4,5,6,7,8,9,10,11,12]

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