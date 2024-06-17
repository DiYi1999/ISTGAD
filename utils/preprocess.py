import argparse
import numpy as np
import torch
import pandas as pd


def preIDW(data):
    df_data = pd.DataFrame(data)
    # (all_len, node_num)
    df_data = df_data.fillna(method="ffill", axis=0)
    df_data = df_data.fillna(method="backfill", axis=0)
    return df_data.values


def preMA(data_array, window_size=50):
    for i in range(data_array.shape[1]):
        for j in range(data_array.shape[0] // window_size):
            start_index = j * window_size
            end_index = (j + 1) * window_size
            mean = np.mean(data_array[start_index:end_index, i])
            data_array[start_index:end_index, i] = mean
        # 最后一个窗口的处理
        start_index = (data_array.shape[0] // window_size) * window_size
        end_index = data_array.shape[0]
        mean = np.mean(data_array[start_index:end_index, i])
        data_array[start_index:end_index, i] = mean

    return data_array