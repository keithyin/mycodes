import tensorflow as tf
from models import puregan
from config import cnf
import os
import numpy as np
from tools import visualization
from data import bulid_record
import matplotlib.pyplot as plt
from config.cnf import data_dir as DATADIR

config_ = cnf.Gun_Point


def show_true_data(file_path):
    datas, clas, feat_len, _, _ = bulid_record.build_dataset(file_path)
    ready_to_show = []
    label_id = []
    for label, data in datas:
        if label not in label_id:
            label_id.append(label)
            ready_to_show.append(data)

    visualization.show(np.array(ready_to_show))


def main():
    dataset_name = config_.DATASET_NAME
    data_dir = os.path.join(DATADIR, dataset_name)

    train_dataset_name = os.path.join(data_dir, dataset_name + "_best10-2_TRAIN")
    test_dataset_name = os.path.join(data_dir, dataset_name + "_TEST")
    show_true_data(test_dataset_name)
    # build TSCGAN

    model = puregan.GAN(config=config_)

    fake_data = model.G(training=False)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model.restore_from_latest_ckpt(ckpt_dir='ckpt_pure_gan/')
        f_d = sess.run(fake_data)
        f_d = np.squeeze(f_d, axis=-1)
        visualization.show(f_d)
    plt.show()


if __name__ == '__main__':
    main()
    pass
