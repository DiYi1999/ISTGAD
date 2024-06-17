import numpy as np
from scipy.stats import iqr
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


def robust_normalize(anomaly_score_tensor):
    anomaly_score_array = anomaly_score_tensor.cpu().numpy()
    for i in range(anomaly_score_tensor.shape[1]):
        median_score = np.median(anomaly_score_array[:, i])
        iqr_score = iqr(anomaly_score_array[:, i])
        anomaly_score_array[:, i] = (anomaly_score_array[:, i] - median_score) / (iqr_score + 1e-8)
    normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)

    return normal_anomaly_score_tensor


def moving_average(anomaly_score_tensor, window_num):
    anomaly_score_array = anomaly_score_tensor.cpu().numpy()
    for i in range(anomaly_score_tensor.shape[1]):
        anomaly_score_array[:, i] = np.convolve(anomaly_score_array[:, i], np.ones(window_num) / window_num, mode='same')
    normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)

    return normal_anomaly_score_tensor


def all_normalize(anomaly_score_tensor):
    anomaly_score_array = anomaly_score_tensor.cpu().numpy()
    median_score = np.median(anomaly_score_array)
    iqr_score = iqr(anomaly_score_array)
    if iqr_score == 0:
        anomaly_score_array = (anomaly_score_array - np.min(anomaly_score_array)) \
                              / (np.max(anomaly_score_array) - np.min(anomaly_score_array) + 1e-8)
    else:
        anomaly_score_array = (anomaly_score_array - median_score) / (iqr_score + 1e-8)
    normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)

    return normal_anomaly_score_tensor


def get_best_performance_data(total_err_scores, gt_labels, topk=1, focus_on='F1', thresold=None):
    total_err_scores = np.transpose(total_err_scores)
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    acc = accuracy_score(gt_labels, pred_labels)
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    print(f'f1,acc,pre,rec:{f1, acc, pre, rec}')

    return f1, acc, pre, rec, auc_score, thresold


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe