import pandas as pd
import numpy as np
from numpy import percentile
import tqdm
from model.utils import get_metrics

def bootstrap_resample(labels, preds, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(labels)
    resample_i = np.floor(np.random.rand(n)*len(labels)).astype(int)
    labels_resample = labels[resample_i]
    preds_resample = preds[resample_i]
    return labels_resample, preds_resample

def get_ci(scores, confidence=95.0):
    
    alpha = 100-confidence
    # calculate lower percentile
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = round(max(0.0, percentile(scores, lower_p)), 3)
    # calculate upper percentile
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = round(min(1.0, percentile(scores, upper_p)), 3)
    return [lower, upper]

def boostrap_ci(labels, preds, metrics_dict, n_boostrap=10000, thresh_val=0.5, csv_path=None, confidence=95.0):
    # columns = df.columns
    metrics_score = dict.fromkeys(metrics_dict.keys(), 0.0)

    for i in tqdm.tqdm(range(n_boostrap)):
        labels_resample, preds_resample = bootstrap_resample(labels, preds)
        # labels = torch.Tensor(np.stack([df_resample[class_name].values for class_name in columns[1:]], axis=-1)).float()
        # preds = torch.Tensor(np.stack([df_resample_pred[class_name].values for class_name in columns[1:]], axis=-1)).float()
        running_metrics = get_metrics(preds_resample, labels_resample, metrics_dict, thresh_val)

        for key in running_metrics.keys():
            if key != 'loss':
                if i == 0:
                    metrics_score[key] = [running_metrics[key].mean()]
                else:
                    metrics_score[key].append(running_metrics[key].mean())
    metrics_score.pop('loss')
    ci_dict = dict.fromkeys(metrics_score.keys(), 0.0)

    for key in metrics_score.keys():
        ci_dict[key] = get_ci(metrics_score[key], confidence=confidence)
    
    if csv_path is not None:
        df_ci = pd.DataFrame.from_dict(ci_dict)
        df_ci.to_csv(csv_path, index=False)
    
    return ci_dict