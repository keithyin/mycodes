"""


"""
import torch
from torch.autograd import Variable
import progressbar
import os
import sys
from data.read_data_pyt import get_loader
from data.read_data_pyt import get_dataset_val
from nets.models import MixModel
from utils import metrics

Proj_Dir = sys.path[0]

GLOBAL_STEP = 0
EPOCH = 0

BATCH_SIZE = 10
SEQ_LEN = 2


def preprocess_batch_data(data, seq_len):
    """
    [batch_size, seq+1, 3, 299, 299] to
    [batch_size, 3, 299, 299] and [batch_size*seq, 3, 299, 299]
    :param data:
    :return:
    """
    batch_size = len(data)
    assert len(data[0]) == (seq_len + 1)
    res_batch_data_shape = (batch_size, data.size(2), data.size(3), data.size(4))
    res_batch_seq_data_shape = (batch_size * seq_len, data.size(2), data.size(3), data.size(4))

    batch_data = []
    batch_seq_data = []
    for i in range(batch_size):
        batch_data.append(data[i][-1])
        batch_seq_data.append(data[i][:-1])
    batch_data = torch.stack(batch_data, dim=0)
    batch_seq_data = torch.cat(batch_seq_data, dim=0)

    assert batch_data.size() == res_batch_data_shape
    assert batch_seq_data.size() == res_batch_seq_data_shape
    return batch_data, batch_seq_data


def validation(queries_loader, gallery_loader, model, query_test_count):
    model.eval()
    widgets = ["processing: ", progressbar.Percentage(),
               " ", progressbar.ETA(),
               " ", progressbar.FileTransferSpeed(),
               ]
    bar = progressbar.ProgressBar(widgets=widgets,
                                  max_value=len(queries_loader) + len(gallery_loader)).start()

    """
    1. calculate all the queries feature and get corresponding label 
    2. calculate gallery feature and get corresponding label
    3. 
    """
    queries_feature = []
    queries_label = []
    gallery_feature = []
    gallery_label = []

    for i, batch in enumerate(queries_loader):
        bar.update(i)
        mix_batch_data, batch_label = batch

        batch_data, batch_seq_data = preprocess_batch_data(mix_batch_data,
                                                           seq_len=SEQ_LEN)

        batch_data = Variable(batch_data, volatile=True).type(torch.FloatTensor).cuda()
        batch_seq_data = Variable(batch_seq_data).type(torch.FloatTensor).cuda()

        batch_label = Variable(batch_label).type(torch.LongTensor).cuda()
        logits = model(batch_data, batch_seq_data)
        queries_feature.append(logits)
        queries_label.append(batch_label)

    # queries_feature [num_queries, feature_size]
    queries_feature = torch.cat(queries_feature, dim=0)

    # queries_feature [num_gallery_samples, feature_size]
    queries_label = torch.cat(queries_label, dim=0)

    for i, batch in enumerate(gallery_loader):
        bar.update(len(queries_loader) + i)
        mix_batch_data, batch_label = batch

        batch_data, batch_seq_data = preprocess_batch_data(mix_batch_data,
                                                           seq_len=SEQ_LEN)

        batch_data = Variable(batch_data, volatile=True).type(torch.FloatTensor).cuda()
        batch_seq_data = Variable(batch_seq_data).type(torch.FloatTensor).cuda()

        batch_label = Variable(batch_label).type(torch.LongTensor).cuda()
        logits = model(batch_data, batch_seq_data)
        gallery_feature.append(logits)
        gallery_label.append(batch_label)

    bar.finish()

    gallery_feature = torch.cat(gallery_feature, dim=0)
    gallery_label = torch.cat(gallery_label, dim=0)

    print("num_queries", len(queries_label))
    print("num gallery samples: ", len(gallery_label))

    print("calculating metrics")
    metrics.metrics(queries=queries_feature, gallery_features=gallery_feature,
                    queries_label=queries_label, gallery_label=gallery_label,
                    query_test_count=query_test_count, n=10)


def main():
    # get dataset

    queries_dataset, gallery_dataset = \
        get_dataset_val(root=
                        "/media/fanyang/workspace/DataSet/MARS/bbox_test",
                        txt_file=os.path.join(Proj_Dir, 'data/id2folderVal.txt'), seq_len=SEQ_LEN)

    # prepare model
    model = MixModel(num_classes=625, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
                     transform_input=True)

    model.load_state_dict(
        torch.load("/home/fanyang/PycharmProjects/PersonReID_CL/ckptseq/model-v1.pkl"))
    model.cuda()

    queries_loader = get_loader(dataset=queries_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, drop_last=False)
    gallery_loader = get_loader(dataset=gallery_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, drop_last=False)

    query_test_count = gallery_dataset.get_query_test_count()

    validation(queries_loader=queries_loader, gallery_loader=gallery_loader, model=model,
               query_test_count=query_test_count)


if __name__ == '__main__':
    main()
