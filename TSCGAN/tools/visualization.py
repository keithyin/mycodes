import matplotlib.pyplot as plt
import numpy as np


def show(datas):
    assert isinstance(datas, np.ndarray)
    num_datas = datas.shape[0]
    n_rows = np.ceil(np.sqrt(num_datas)).astype(np.int64)
    n_cols = n_rows

    fig, axes = plt.subplots(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            data_index = i*n_cols+j
            if data_index >= num_datas:
                break
            axes[i, j].plot(datas[data_index])


