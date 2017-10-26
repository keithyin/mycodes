"""
this file is used to calculate the metrics

MAP, Precision, Recall

"""
import time
import torch
from torch import cuda
from torch.autograd import Variable


def metrics(queries, gallery_features, queries_label, gallery_label, query_test_count, n):
    """
    
    :param queries:  [query_size, feature_size] the features of queries
    :param gallery_features:  [gallery_size, feature_size]  all the features in the gallery
    :param queries_label: corresponding with the queries, specify the label of the queries
    :param gallery_label: corresponding with the gallery_features,
    specify the label of the gallery_features
    :param query_test_count: store the number of positive sample in the gallery,
    corresponding to queries
    :return: None
    """
    start = time.time()

    assert isinstance(queries, (Variable, torch.FloatTensor, cuda.FloatTensor))
    assert isinstance(gallery_features, (Variable, torch.FloatTensor, cuda.FloatTensor))
    assert isinstance(queries_label, (Variable, torch.LongTensor, cuda.LongTensor))
    assert isinstance(gallery_label, (Variable, torch.LongTensor, cuda.LongTensor))
    if isinstance(queries, Variable):
        queries = queries.data
    if isinstance(gallery_features, Variable):
        gallery_features = gallery_features.data
    if isinstance(queries_label, Variable):
        queries_label = queries_label.data
        assert isinstance(queries_label, (torch.LongTensor, cuda.LongTensor))
    if isinstance(gallery_label, Variable):
        gallery_label = gallery_label.data
        assert isinstance(gallery_label, (torch.LongTensor, cuda.LongTensor))

    normalized_queries_features = \
        queries / torch.unsqueeze(torch.norm(source=queries, dim=1, p=2.), dim=1)

    normalized_gallery_features = \
        gallery_features / torch.unsqueeze(torch.norm(source=gallery_features, p=2., dim=1), dim=1)
    print('num queries:', len(normalized_queries_features))
    print('num gallery samples:', len(normalized_gallery_features))

    # [num_queries, num_gallery_sample]
    cos_simi = torch.matmul(normalized_queries_features, torch.transpose(normalized_gallery_features,
                                                                         dim0=0, dim1=1))

    # retrieval [num_queries, num_gallery_samples]
    _, retrieval = torch.sort(cos_simi, dim=1, descending=True)

    precision = cal_precision(retrieval=retrieval,
                              queries_label=queries_label,
                              gallery_label=gallery_label,
                              n=n)
    recall = cal_recall(retrieval=retrieval, queries_label=queries_label,
                        gallery_label=gallery_label,
                        n=n, query_test_count=query_test_count)

    mean_average_precision = cal_mean_average_precision(retrieval=retrieval,
                                                        queries_label=queries_label,
                                                        gallery_label=gallery_label,
                                                        n=len(gallery_label))

    print("precision:{}, recall:{}, map:{}".format(precision, recall, mean_average_precision))
    print('evaluation time:', time.time() - start)


def cal_precision(retrieval, queries_label, gallery_label, n):
    """

    :param retrieval: a sorted index of [num_queries, num_gallery_samples]
    :param queries_label: [num_queries]
    :param gallery_label: [num_gallery_samples]
    :param n:
    :return:
    """
    eql = in_first_n_results(retrieval=retrieval, queries_label=queries_label,
                             gallery_label=gallery_label,
                             n=n)
    mean_precision = torch.mean(eql)
    return mean_precision


def in_first_n_results(retrieval, queries_label, gallery_label, n):
    num_queries = len(retrieval)
    # top_n_id [num_queries, n]
    top_n_id = retrieval[:, :n]

    # broadcasted_gallery_label [num_queries, num_gallery_samples]
    broadcasted_gallery_label = torch.unsqueeze(gallery_label, dim=0).expand(num_queries,
                                                                             len(gallery_label))

    # top_n_label [num_queries, n]
    top_n_label = top_n_id.new().resize_as_(top_n_id).zero_()

    # using the top n id to gather the corresponding label
    torch.gather(broadcasted_gallery_label, dim=1, index=top_n_id, out=top_n_label)

    # broadcasted_queries_label [num_queries, n]
    broadcasted_queries_label = torch.unsqueeze(queries_label, dim=1).expand(num_queries, n)
    eql = (top_n_label == broadcasted_queries_label)
    eql = eql.type(cuda.FloatTensor)

    return eql


def cal_recall(retrieval, queries_label, gallery_label, n, query_test_count):
    """

    :param retrieval: []
    :param queries_label:
    :param gallery_label:
    :param n:
    :param query_test_count:
    :return:
    """
    eql = in_first_n_results(retrieval=retrieval, queries_label=queries_label,
                             gallery_label=gallery_label,
                             n=n)
    # tp : true positive
    tp = torch.sum(eql, dim=1)

    recall_ = tp / query_test_count.type(cuda.FloatTensor)
    mean_recall = torch.mean(recall_)
    return mean_recall


def cal_mean_average_precision(retrieval, queries_label, gallery_label, n):
    num_queries = len(retrieval)
    eql = in_first_n_results(retrieval=retrieval, queries_label=queries_label,
                             gallery_label=gallery_label,
                             n=n)
    # cumulative_sum [num_queries, n]

    cumulative_sum = torch.cumsum(eql, dim=1)
    numerator = eql * cumulative_sum

    denominator = torch.unsqueeze(torch.arange(1, end=n + 1), dim=0).expand(num_queries, n).cuda()

    mean_average_precision = torch.mean(numerator / denominator)
    return mean_average_precision


def gather_demo():
    id = torch.LongTensor([[1, 2, 3], [0, 1, 2]])
    label = torch.LongTensor([0, 1, 2, 4])
    broadcasted_label = torch.unsqueeze(label, dim=0).expand(2, len(label))
    res = torch.gather(broadcasted_label, dim=1, index=id)
    print(res)


def accuracy(logits, targets):
    """
    cal the accuracy of the predicted result
    :param logits: Variable [batch_size, num_classes]
    :param targets: Variable [batch_size]
    :return: Variable scalar
    """
    assert isinstance(logits, Variable)
    val, idx = logits.max(dim=1)
    eql = (idx == targets)
    eql = eql.type(torch.cuda.FloatTensor)
    res = torch.mean(eql)
    return res


def main():
    gather_demo()


if __name__ == '__main__':
    main()
