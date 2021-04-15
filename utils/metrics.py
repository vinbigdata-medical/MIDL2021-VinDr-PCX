import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


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

    def forward(self, pred, target, thresholding=False):
        p_n = pred.cpu().detach().numpy()
        t_n = target.cpu().detach().numpy()
        n_classes = pred.shape[-1]
        auclist = []
        list_threshold = []
        for i in range(n_classes):
            # print(i)
            fpr, tpr, thresholds = roc_curve(t_n[:, i], p_n[:, i])
            if thresholding:
                J = tpr - fpr
                ix = np.argmax(J)
                list_threshold.append(thresholds[ix])
            auclist.append(auc(fpr, tpr))
        auc_score = torch.from_numpy(np.array(auclist))
        if thresholding:
            return list_threshold
        return auc_score