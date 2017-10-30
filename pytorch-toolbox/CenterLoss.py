import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class CenterLoss(nn.Module):
    def __init__(self, feature_len, num_classes, decay_rate=.99):
        super(CenterLoss, self).__init__()
        self.register_buffer('center_feature', torch.zeros(num_classes, feature_len))
        self.decay_rate = decay_rate
        self.feature_len = feature_len

        self.register_buffer('all_labels',
                             torch.arange(
                                 start=0., end=float(num_classes)).type(torch.LongTensor))

        # we can change the criterion
        # TODO: using other criterion to test the result
        self.criterion = nn.MSELoss()

    def forward(self, batch_feature, batch_label):
        """
        calculate the CenterLoss
        :param batch_feature: Float Variable, [batch, feature_len]
        :param batch_label: Long Variable, [batch]
        :return: center loss
        """
        self.__update_center_feature(batch_feature, batch_label)

        return self.__center_loss(batch_feature, batch_label)

    def __center_loss(self, batch_feature, batch_label):
        batch_label = batch_label.data
        batch_size = len(batch_label)

        batch_centen_f = [self.center_feature[batch_label[i]] for i in range(batch_size)]
        batch_centen_f = Variable(torch.stack(batch_centen_f, dim=0))
        loss = self.criterion(batch_feature, batch_centen_f)
        return loss

    def __update_center_feature(self, batch_feature, batch_label):
        """
        1. how to mean the given label's feature (????)
        2. how to mark given label (can do)
        """
        batch_feature = batch_feature.data
        batch_label = batch_label.data

        for label in self.all_labels:
            has_label = (label == batch_label)
            """
            how to mean the batch features given label
            1. gather the features given label from batch_feature
            2. mean them
            """
            mask = torch.unsqueeze(has_label, dim=1).expand_as(batch_feature)
            if torch.sum(has_label) == 0:
                continue

            masked_feature = torch.masked_select(batch_feature, mask)
            masked_feature = masked_feature.view(-1, self.feature_len)
            mean_masked_feature = torch.mean(masked_feature, dim=0)
            if not self.all_feature_is_initialized:
                if label not in self.initialized_label:
                    self.initialized_label.append(label)
                    self.center_feature[label] = mean_masked_feature

                    if len(self.initialized_label) == self.num_classes:
                        self.all_feature_is_initialized = True

                    continue
            self.center_feature[label] = \
                self.decay_rate * self.center_feature[label] + \
                (1. - self.decay_rate) * \
                mean_masked_feature


def main():
    cl = CenterLoss(feature_len=10, num_classes=10)
    features = Variable(torch.randn(5, 10))
    labels = Variable(torch.LongTensor([2, 1, 3, 2, 0]))
    print(cl(features, labels))


if __name__ == '__main__':
    main()
