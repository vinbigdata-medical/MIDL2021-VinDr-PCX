from easydict import EasyDict as edict
import json, os
from utils.metrics import F1, AUC, Specificity, Recall
from torch.nn import BCELoss, BCEWithLogitsLoss
from model.classifier import Pediatric_Classifier
from data.dataset import create_loader
import warnings

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

finetune = False

# cfg_path = './config/chexmic_config.json'
cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

data_dir = '/home/tungthanhlee/thanhtt/assigned_jpeg'

train_loader = create_loader(cfg.train_csv, data_dir, cfg, mode='train')
val_loader = create_loader(cfg.dev_csv, data_dir, cfg, mode='val')

loss_func = BCEWithLogitsLoss()

metrics_dict = {'auc':AUC(), 'sensitivity':Recall(), 'specificity':Specificity(), 'f1':F1()}
loader_dict = {'train': train_loader, 'val': val_loader}

pediatric_classifier = Pediatric_Classifier(cfg, loss_func, metrics_dict)
if finetune and os.path.isfile(cfg.ckp_path):
    pediatric_classifier.load_backbone(cfg.ckp_path, strict=False)

if not os.path.exists('experiment'):
    os.makedirs('experiment')
os.makedirs(os.path.join('experiment', cfg.log_dir), exist_ok=True)
ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

pediatric_classifier.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, eval_metric='auc', ckp_dir=ckp_dir, resume=False)

if cfg.type == 'chexmic':
    pediatric_classifier.save_backbone(os.path.join(ckp_dir, 'backbone.ckpt'))