from easydict import EasyDict as edict
import json, os

from torch._C import ErrorReport
from utils.metrics import F1, AUC, Specificity, Recall
from torch.nn import BCELoss, BCEWithLogitsLoss
from model.classifier import Pediatric_Classifier
from data.dataset import create_loader
import warnings
import argparse

# warnings.simplefilter('always')
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--config', type=str, default='config/example.json', help='*.config path')
    parser.add_argument('--finetune', action='store_true', help='finetuning from pre-trained weights on CheXpert and Mimic dataset')

    opt = parser.parse_args()
    print(opt)

    with open(opt.config) as f:
        cfg = edict(json.load(f))

    train_loader = create_loader(cfg.train_csv, cfg, mode='train')
    val_loader = create_loader(cfg.dev_csv, cfg, mode='val')

    loss_func = BCEWithLogitsLoss()

    metrics_dict = {'auc':AUC(), 'sensitivity':Recall(), 'specificity':Specificity(), 'f1':F1()}

    pediatric_classifier = Pediatric_Classifier(cfg, loss_func, metrics_dict)
    if opt.finetune:
        if os.path.isfile(cfg.ckp_path):
            pediatric_classifier.load_backbone(cfg.ckp_path, strict=False)
        else:
            raise ErrorReport("pre-trained path not defined!!!")

    if not os.path.exists('experiment'):
        os.makedirs('experiment')
    os.makedirs(os.path.join('experiment', cfg.log_dir), exist_ok=True)
    ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

    pediatric_classifier.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, eval_metric='auc', ckp_dir=ckp_dir, resume=False)

    if cfg.type == 'chexmic':
        pediatric_classifier.save_backbone(os.path.join(ckp_dir, 'backbone.ckpt'))