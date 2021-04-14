from itertools import count
import numpy as np
from numpy.lib.function_base import average
from numpy.lib.npyio import save
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os
from collections import OrderedDict

disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity", "Diffuse aveolar opacity", "Lung hyperinflation",
                   "Consolidation", "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]

def get_loss(output, target, index, device, cfg):
    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1), target, pos_weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=pos_weight[index])

    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return loss


class OA_loss(nn.Module):

    def __init__(self, device, cfg):
        """modify loss funtion

        Args:
            device (torch.device): device
            cfg (dict): configuration file.
        """
        super(OA_loss, self).__init__()
        self.device = device
        self.cfg = cfg

    def forward(self, pred, target):
        num_tasks = len(self.cfg.num_classes)
        loss_sum = 0.0
        for t in range(num_tasks):
            loss_t = get_loss(pred, target, t, self.device, self.cfg)
            loss_sum += loss_t*(1/num_tasks)
        return loss_sum


class Precision(nn.Module):

    def __init__(self, eps=1e-5):
        """Precision score

        Args:
            thresh (float, optional): threshold value. Defaults to 0.5.
            eps (float, optional): epsilon value to prevent zero division. Defaults to 1e-5.
        """
        super(Precision, self).__init__()
        self.eps = eps

    def forward(self, pred, target, thresh=0.5):
        pred_thresh = (pred > thresh)*1.0
        tp = torch.sum((pred_thresh == target)*(target == 1.0), dim=0)
        fp = torch.sum((pred_thresh != target)*(target == 0.0), dim=0)
        precision = tp / (fp+tp+self.eps)
        return precision


class Recall(nn.Module):

    def __init__(self, eps=1e-5):
        """Recall score

        Args:
            thresh (float, optional): threshold value. Defaults to 0.5.
            eps (float, optional): epsilon value to prevent zero division. Defaults to 1e-5.
        """
        super(Recall, self).__init__()
        self.eps = eps

    def forward(self, pred, target, thresh=0.5):
        pred_thresh = (pred > thresh)*1.0
        tp = torch.sum((pred_thresh == target)*(target == 1.0), dim=0)
        fn = torch.sum((pred_thresh != target)*(target == 1.0), dim=0)
        recall = tp / (fn+tp+self.eps)
        return recall


class Specificity(nn.Module):

    def __init__(self, eps=1e-5):
        """F1 score

        Args:
            thresh (float, optional): threshold value. Defaults to 0.5.
            eps (float, optional): epsilon value to prevent zero division. Defaults to 1e-5.
        """
        super(Specificity, self).__init__()
        self.eps = eps

    def forward(self, pred, target, thresh=0.5):
        pred_thresh = (pred > thresh)*1.0
        # tp = torch.sum((pred_thresh == target)*(target==1.0), dim=0)
        tn = torch.sum((pred_thresh == target)*(target == 0.0), dim=0)
        fp = torch.sum((pred_thresh != target)*(target == 0.0), dim=0)
        specificity = tn / (fp+tn+self.eps)
        return specificity


class F1(nn.Module):

    def __init__(self, eps=1e-5):
        """F1 score

        Args:
            thresh (float, optional): threshold value. Defaults to 0.5.
            eps (float, optional): epsilon value to prevent zero division. Defaults to 1e-5.
        """
        super(F1, self).__init__()
        self.eps = eps

    def forward(self, pred, target, thresh=0.5):
        pred_thresh = (pred > thresh)*1.0
        tp = torch.sum((pred_thresh == target)*(target == 1.0), dim=0)
        fp = torch.sum((pred_thresh != target)*(target == 0.0), dim=0)
        fn = torch.sum((pred_thresh != target)*(target == 1.0), dim=0)
        recall = tp / (fn+tp+self.eps)
        precision = tp / (fp+tp+self.eps)
        f1_score = 2 * precision * recall / (precision + recall + self.eps)
        return f1_score


class ACC(nn.Module):

    def __init__(self):
        """ACC score

        Args:
            thresh (float, optional): threshold value. Defaults to 0.5.
        """
        super(ACC, self).__init__()

    def forward(self, pred, target, thresh=0.5):
        pred_thresh = (pred > thresh)*1.0
        t = torch.sum(pred_thresh == target, dim=0)
        # print(t)
        return t/(float(pred.shape[0]))


class AUC(nn.Module):
    def __init__(self):
        """AUC score
        """
        super(AUC, self).__init__()

    def forward(self, pred, target):
        p_n = pred.cpu().detach().numpy()
        t_n = target.cpu().detach().numpy()
        n_classes = pred.shape[-1]
        auclist = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(t_n[:, i], p_n[:, i])
            auclist.append(auc(fpr, tpr))
        auc_score = torch.from_numpy(np.array(auclist))
        return auc_score

# class AUC(nn.Module):
#     def __init__(self):
#         """AUC score
#         """
#         super(AUC, self).__init__()
#     def forward(self, pred, target):
#         p_n = pred.cpu().detach().numpy()
#         t_n = target.cpu().detach().numpy()
#         auc_score = roc_auc_score(t_n, p_n, average = None)
#         auc_score = torch.from_numpy(np.array(auc_score))
#         return auc_score


class AUC_ROC(nn.Module):
    def __init__(self):
        """AUC score
        """
        super(AUC_ROC, self).__init__()

    def forward(self, pred, target, thresholding=False, save=False):
        p_n = pred.cpu().detach().numpy()
        t_n = target.cpu().detach().numpy()
        n_classes = pred.shape[-1]
        color = np.random.randint(0, 256, (n_classes, 3), dtype=np.uint8)
        auclist = []
        # save_path = 'roc_curve'
        count = 0
        save_path = 'ROC_curve_'+str(count)+'.png'
        list_threshold = []
        while os.path.exists(save_path):
            count += 1
            save_path = 'ROC_curve_'+str(count)+'.png'
        plt.figure(figsize=(8, 8))
        plt.title('Receiver Operating Characteristic')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        for i in range(n_classes):
            # print(i)
            fpr, tpr, thresholds = roc_curve(t_n[:, i], p_n[:, i])
            if thresholding:
                J = tpr - fpr
                ix = np.argmax(J)
                list_threshold.append(thresholds[ix])
            auclist.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, color[i], label='{}: {:.2f}'.format(
                disease_classes[i], auclist[-1]))

        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if save:
            plt.savefig(save_path)
        auc_score = torch.from_numpy(np.array(auclist))
        if thresholding:
            return list_threshold
        return auc_score
