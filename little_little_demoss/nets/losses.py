import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self, feature_len, num_classes, decay_rate=.99):
        super(CenterLoss, self).__init__()

        self.num_classes = num_classes
        self.register_buffer('center_feature', torch.zeros(num_classes, feature_len))
        self.decay_rate = decay_rate
        self.feature_len = feature_len

        self.register_buffer('all_labels',
                             torch.arange(
                                 start=0.,
                                 end=float(self.num_classes)).type(torch.LongTensor))

        # we can change the criterion
        # TODO: using other criterion to test the result
        self.criterion = nn.MSELoss()

        self.initialized_label = []
        self.all_feature_is_initialized = False

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


class ImprovedCenterLoss(nn.Module):
    def __init__(self, feature_len, num_classes, decay_rate=.99):
        super(ImprovedCenterLoss, self).__init__()

        self.num_classes = num_classes
        self.register_buffer('center_feature', torch.zeros(num_classes, feature_len))
        self.decay_rate = decay_rate
        self.feature_len = feature_len

        self.register_buffer('all_labels',
                             torch.arange(
                                 start=0.,
                                 end=float(self.num_classes)).type(torch.LongTensor))

        # we can change the criterion
        # TODO: using other criterion to test the result
        self.criterion = nn.MSELoss()

        self.initialized_label = []
        self.all_feature_is_initialized = False

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
                    # normalize the feature space
                    self.center_feature[label] = \
                        self.center_feature[label] / torch.norm(self.center_feature[label], p=2.)

                    if len(self.initialized_label) == self.num_classes:
                        self.all_feature_is_initialized = True

                    continue
            self.center_feature[label] = \
                self.decay_rate * self.center_feature[label] + \
                (1. - self.decay_rate) * \
                mean_masked_feature
            self.center_feature[label] = \
                self.center_feature[label] / torch.norm(self.center_feature[label], p=2.)


class SiameseLoss(nn.Module):
    def __init__(self, threshold=1.):
        super(SiameseLoss, self).__init__()
        self.threshold = threshold

    def forward(self, x, label):
        """
        the forward procedure of siamese loss
        :param x: [batch_size, feature], batch_size % 2 == 0
        :param label: [batch_size]
        :return: siamese loss
        """
        assert len(x) % 2 == 0, "len(x) % 2 must be zero"
        half_batch_size = len(label) // 2
        same_identity = (label[:half_batch_size] == label[half_batch_size:])

        total_loss = Variable(torch.zeros(1).type(type(x.data)))

        for i in range(half_batch_size):
            is_same = (same_identity[i] == 0).data.cpu().numpy()[0]
            if is_same:
                loss = torch.sum((x[i] - x[half_batch_size + i]) ** 2)

                total_loss = total_loss + loss

            else:
                loss = F.threshold(
                    (self.threshold -
                     torch.norm(x[i] - x[half_batch_size + i] ** 2, p=2.)),
                    threshold=0., value=0.)

                total_loss = total_loss + loss

        total_loss = total_loss / (half_batch_size * 2 * 2)
        return total_loss


def main():
    siamese = SiameseLoss()
    inputs = Variable(torch.randn(10, 30).cuda(), requires_grad=True)

    labels = Variable(torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).cuda()
    loss = siamese(inputs, labels)
    loss.backward()
    print(inputs.grad)


if __name__ == '__main__':
    main()
