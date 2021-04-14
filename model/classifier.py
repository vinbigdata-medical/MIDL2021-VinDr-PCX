from numpy.core.shape_base import stack
from numpy.lib.stride_tricks import broadcast_to
import torch.nn as nn
import numpy as np
import time
import cv2
import os
import shutil
from torch.nn.functional import threshold
import tqdm
import pickle
import torch
import wandb
from utils.metrics import AUC_ROC
from data.utils import transform
from model.models import Stacking, save_dense_backbone, load_dense_backbone, save_resnet_backbone, load_resnet_backbone, Ensemble, AverageMeter
from model.utils import get_models, get_str, tensor2numpy, get_optimizer, load_ckp, lrfn, get_metrics, get_device
from utils.confidence_interval import boostrap_ci


class Pediatric_Classifier():

    def __init__(self, cfg, loss_func, metrics=None):
        """Pediatric_Classifier class used to train and evaluate model performance

        Args:
            cfg: contain configuration.
            loss_func: Loss function.
            metrics (dict, optional): dictionary contains evaluation metrics. Defaults to None.
        """
        self.cfg = cfg

        if self.cfg.type == 'pediatric':
            self.cfg.num_classes = 13*[1]
        elif self.cfg.type == 'chexmic':
            self.cfg.num_classes = 14*[1]
        else:
            self.cfg.num_classes = [1]
        self.device = get_device(self.cfg.device)
        self.model = get_models(self.cfg)

        if self.cfg.ensemble == 'stacking':
            self.stacking_model = Stacking(len(self.model))
            if os.path.isfile(self.cfg.ckp_stack):
                self.stacking_model.load_state_dict(torch.load(
                    self.cfg.ckp_stack, map_location=torch.device("cpu")))
                self.stacking_model.to(self.device)
            self.stacking_model.freeze()
        self.loss_func = loss_func

        if metrics is not None:
            self.metrics = metrics
            self.metrics['loss'] = self.loss_func
        else:
            self.metrics = {'loss': self.loss_func}

        if cfg.parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.thresh_val = torch.Tensor(
            [0.5]*len(self.cfg.num_classes)).float().to(self.device)

    def save_backbone(self, ckp_path):
        """
        Save model backbone to ckp_path.
        """
        if self.cfg.parallel:
            model_part = self.model.module
        else:
            model_part = self.model

        if self.cfg.backbone == 'dense' or self.cfg.backbone == 'densenet':
            save_dense_backbone(model_part, ckp_path)
        elif self.cfg.backbone == 'resnet':
            save_resnet_backbone(model_part, ckp_path)

    def load_backbone(self, ckp_path, strict=True):
        """
        Load model backbone to ckp_path.
        """
        if self.cfg.parallel:
            model_part = self.model.module
        else:
            model_part = self.model

        if self.cfg.backbone == 'dense' or self.cfg.backbone == 'densenet':
            load_dense_backbone(model_part, ckp_path, self.device, strict)
        elif self.cfg.backbone == 'resnet':
            load_resnet_backbone(model_part, ckp_path, self.device, strict)

    def load_ckp(self, ckp_path, strict=True):
        """
        Load model from ckp_path. 
        """
        return load_ckp(self.model, ckp_path, self.device, self.cfg.parallel, strict)

    def save_ckp(self, ckp_path, epoch, iter):
        """
        Save model to ckp_path.
        """
        if os.path.exists(os.path.dirname(ckp_path)):
            torch.save(
                {'epoch': epoch+1,
                 'iter': iter+1,
                 'state_dict': self.model.module.state_dict() if self.cfg.parallel else self.model.state_dict()},
                ckp_path
            )
        else:
            print("Save path not exist!!!")

    def predict(self, image):
        """Run prediction

        Args:
            image (torch.Tensor): images to predict. Shape (batch size, C, H, W)

        Returns:
            torch.Tensor: model prediction
        """
        self.model.eval()
        with torch.no_grad() as tng:
            preds = self.model(image)
            if not isinstance(self.model, Ensemble) and self.cfg.ensemble == 'none':
                preds = nn.Sigmoid()(preds)
            elif self.cfg.ensemble == 'average':
                preds = preds.mean(-1)
            elif self.cfg.ensemble == 'stacking':
                preds = self.stacking_model(preds)

        return preds

    def predict_from_file(self, image_file):
        """Run prediction from image path

        Args:
            image_file (str): image path

        Returns:
            numpy array: model prediction in numpy array type
        """
        image_gray = cv2.imread(image_file, 0)
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return tensor2numpy(nn.Sigmoid()(self.predict(image)))

    def predict_loader(self, loader, cal_loss=False):
        """Run prediction on a given dataloader.

        Args:
            loader (torch.utils.data.Dataloader): a dataloader
            cal_loss (bool): whether to calculate the loss. Defaults to True

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: prediction, labels, loss value
        """
        preds_stack = []
        labels_stack = []
        running_loss = AverageMeter()
        ova_len = loader.dataset.n_data

        loop = tqdm.tqdm(enumerate(loader), total=len(loader))
        for i, data in loop:
            imgs, labels = data[0].to(self.device), data[1].to(self.device)

            if self.cfg.tta:
                # imgs = torch.cat(imgs, dim=0)
                list_imgs = [imgs[:, j] for j in range(imgs.shape[1])]
                imgs = torch.cat(list_imgs, dim=0)
                preds = self.predict(imgs)
                batch_len = labels.shape[0]
                list_preds = [preds[batch_len*j:batch_len *
                                    (j+1)] for j in range(len(list_imgs))]
                preds = torch.stack(list_preds, dim=0).mean(dim=0)
            else:
                preds = self.predict(imgs)
            preds_stack.append(preds)
            labels_stack.append(labels)

            if cal_loss:
                # running_loss.append(self.metrics['loss'](
                #     preds, labels).item()*iter_len/ova_len)
                running_loss.update(self.metrics['loss'](
                    preds, labels).item(), imgs.shape[0])

        preds_stack = torch.cat(preds_stack, 0)
        labels_stack = torch.cat(labels_stack, 0)

        return preds_stack, labels_stack, running_loss.avg

    def train(self, train_loader, val_loader, epochs=10, iter_log=100, use_lr_sch=False, resume=False, ckp_dir='./experiment/checkpoint',
              eval_metric='auc'):
        """Run training

        Args:
            train_loader (torch.utils.data.Dataloader): dataloader use for training
            val_loader (torch.utils.data.Dataloader): dataloader use for validation
            epochs (int, optional): number of training epochs. Defaults to 120.
            iter_log (int, optional): logging iteration. Defaults to 100.
            use_lr_sch (bool, optional): use learning rate scheduler. Defaults to False.
            resume (bool, optional): resume training process. Defaults to False.
            ckp_dir (str, optional): path to checkpoint directory. Defaults to './experiment/checkpoint'.
            eval_metric (str, optional): name of metric for validation. Defaults to 'loss'.
        """
        wandb.init(name=self.cfg.log_dir,
                   project='Pediatric Multi-label Classifier')

        optimizer = get_optimizer(self.model.parameters(), self.cfg)

        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lrfn)
            lr_hist = []
        else:
            lr_sch = None
        best_metric = 0.0

        if os.path.exists(ckp_dir) != True:
            os.mkdir(ckp_dir)
        if resume:
            epoch_resume, iter_resume = self.load_ckp(
                os.path.join(ckp_dir, 'latest.ckpt'))
        else:
            epoch_resume = 1
            iter_resume = 0
        scaler = None
        if self.cfg.mix_precision:
            print('Train with mix precision!')
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epoch_resume-1, epochs):
            start = time.time()
            running_loss = AverageMeter()
            n_iter = len(train_loader)
            torch.set_grad_enabled(True)
            self.model.train()
            batch_weights = (1/iter_log)*np.ones(n_iter)
            step_per_epoch = n_iter // iter_log
            if n_iter % iter_log:
                step_per_epoch += 1
                batch_weights[-(n_iter % iter_log):] = 1 / (n_iter % iter_log)
                iter_per_step = iter_log * \
                    np.ones(step_per_epoch, dtype=np.int16)
                iter_per_step[-1] = n_iter % iter_log
            else:
                iter_per_step = iter_log * \
                    np.ones(step_per_epoch, dtype=np.int16)
            i = 0
            for step in range(step_per_epoch):
                loop = tqdm.tqdm(
                    range(iter_per_step[step]), total=iter_per_step[step])
                iter_loader = iter(train_loader)
                for iteration in loop:
                    data = next(iter_loader)
                    imgs, labels = data[0].to(
                        self.device), data[1].to(self.device)

                    if self.cfg.mix_precision:
                        with torch.cuda.amp.autocast():
                            preds = self.model(imgs)
                            loss = self.metrics['loss'](preds, labels)

                    else:
                        preds = self.model(imgs)
                        loss = self.metrics['loss'](preds, labels)

                    preds = nn.Sigmoid()(preds)
                    running_loss.update(loss.item(), imgs.shape[0])
                    optimizer.zero_grad()
                    if self.cfg.mix_precision:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    i += 1

                if wandb:
                    wandb.log(
                        {'loss/train': running_loss.avg}, step=(epoch*n_iter)+(i+1))
                s = "Epoch [{}/{}] Iter [{}/{}]:\n".format(
                    epoch+1, epochs, i+1, n_iter)
                s += "{}_{} {:.3f}\n".format('train', 'loss', running_loss.avg)
                running_metrics_test = self.test(
                    val_loader, False)
                torch.set_grad_enabled(True)
                self.model.train()
                s = get_str(running_metrics_test, 'val', s)
                if wandb:
                    for key in running_metrics_test.keys():
                        if key != 'loss':
                            for j, disease_class in enumerate(np.array(train_loader.dataset.disease_classes)):
                                wandb.log(
                                    {key+'/'+disease_class: running_metrics_test[key][j]}, step=(epoch*n_iter)+(i+1))
                        else:
                            wandb.log(
                                {'loss/val': running_metrics_test['loss']}, step=(epoch*n_iter)+(i+1))
                if self.cfg.type != 'chexmic':
                    metric_eval = running_metrics_test[eval_metric]
                else:
                    metric_eval = running_metrics_test[eval_metric][self.id_obs]
                s = s[:-1] + "- mean_"+eval_metric + \
                    " {:.3f}".format(metric_eval.mean())
                self.save_ckp(os.path.join(
                    ckp_dir, 'latest.ckpt'), epoch, i)
                running_loss.reset()
                end = time.time()
                s += " ({:.1f}s)".format(end-start)
                print(s)
                if metric_eval.mean() > best_metric:
                    best_metric = metric_eval.mean()
                    shutil.copyfile(os.path.join(ckp_dir, 'latest.ckpt'), os.path.join(
                        ckp_dir, 'best.ckpt'))
                    print('new checkpoint saved!')
                start = time.time()
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))
        if lr_sch is not None:
            return lr_hist
        else:
            return None

    def test(self, loader, get_ci=False, n_boostrap=10000):
        """Run testing

        Args:
            loader (torch.utils.data.Dataloader): dataloader use for testing.
            get_ci (bool, optional): whether to calculate the confidence interval. Defaults to False.
            n_boostrap (int, optional): number of Bootstrap samples. Defaults to 10000.

        Returns:
            dict: dictionary of evaluated metrics.
        """
        preds_stack, labels_stack, running_loss = self.predict_loader(
            loader, cal_loss=True)

        running_metrics = get_metrics(
            preds_stack, labels_stack, self.metrics, self.thresh_val)
        running_metrics['loss'] = running_loss
        if get_ci:
            ci_dict = self.eval_CI(labels_stack, preds_stack, n_boostrap)
            return running_metrics, ci_dict

        return running_metrics

    def thresholding(self, loader):
        """Run thresholding using Youden's J statistic.

        Args:
            loader (torch.utils.data.Dataloader): dataloader use for thresholding.
        """
        auc_opt = AUC_ROC()
        preds, labels, _ = self.predict_loader(loader)
        thresh_val = auc_opt(preds, labels, thresholding=True)
        print(f"List optimal threshold {thresh_val}")
        self.thresh_val = torch.Tensor(thresh_val).float().cuda()

    def eval_CI(self, labels, preds, n_boostrap=1000, csv_path=None):
        """
        Calculate confidence interval using Bootstrap Sampling.
        """
        return boostrap_ci(labels, preds, self.metrics, n_boostrap, self.thresh_val, csv_path)

    def stacking(self, train_loader, val_loader, epochs=10, eval_metric='auc'):
        """
        Run stacking ensemble.
        """
        if not isinstance(self.model, Ensemble):
            raise Exception("model must be Ensemble!!!")

        optimizer = get_optimizer(self.stacking_model.parameters(), self.cfg)
        def lambda1(epoch): return 0.9 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda1)

        os.makedirs(os.path.join(
            'experiment', self.cfg.log_dir), exist_ok=True)
        ckp_dir = os.path.join('experiment', self.cfg.log_dir, 'checkpoint')
        os.makedirs(ckp_dir, exist_ok=True)

        self.model.freeze()
        self.stacking_model.unfreeze()
        self.stacking_model.cuda()

        running_loss = AverageMeter()
        best_metric = 0.0

        for epoch in range(epochs):
            self.stacking_model.train()
            for i, data in enumerate(tqdm.tqdm(train_loader)):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                preds = self.stacking_model(self.model(imgs))
                loss = self.metrics['loss'](preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.update(loss.item(), imgs.shape[0])
            s = "Epoch [{}/{}]:\n".format(
                epoch+1, epochs)
            s += "{}_{} {:.3f}\n".format('train', 'loss', running_loss.avg)
            self.stacking_model.eval()
            running_metrics = self.test(val_loader)
            running_metrics.pop('loss')
            s = get_str(running_metrics, 'val', s)
            metric_eval = running_metrics[eval_metric]
            s = s[:-1] + "- mean_"+eval_metric + \
                " {:.3f}".format(metric_eval.mean())
            torch.save(self.stacking_model.state_dict(),
                       os.path.join(ckp_dir, 'latest.ckpt'))
            running_loss.reset()
            scheduler.step()
            print(s)
            if metric_eval.mean() > best_metric:
                best_metric = metric_eval.mean()
                shutil.copyfile(os.path.join(ckp_dir, 'latest.ckpt'), os.path.join(
                    ckp_dir, 'best.ckpt'))
                print('new checkpoint saved!')
