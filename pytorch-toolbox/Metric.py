"""
for computing the metric
"""

from torch.autograd import Variable
import torch

# TODO: combine it with routine

class MetricBase(object):
    def __init__(self):
        pass

    def __call__(self, *inputs, **kwargs):
        # make sure that the computation in update metric is tensor manipulation
        inputs, kwargs = MetricBase._variables_to_tensors(*inputs, **kwargs)

        self.update_metric(*inputs, **kwargs)

    def update_metric(self, *inputs, **kwargs):
        # all the things happened here is tensor manipulation
        raise NotImplementedError

    def show_state(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError

    @staticmethod
    def _variables_to_tensors(*inputs, **kwargs):
        new_inputs = []
        new_keyword_inputs = {}
        for val in inputs:
            if isinstance(val, Variable):
                val = val.data
            new_inputs.append(val)
        for key, val in kwargs.items():
            if isinstance(val, Variable):
                val = val.data
            new_keyword_inputs.update({key: val})
        return new_inputs, new_keyword_inputs


class AccuracyMetric(MetricBase):
    def __init__(self):
        super(AccuracyMetric, self).__init__()
        self._accu = 0.
        self.counter = 0

    def update_metric(self, logits, labels):
        assert len(logits) == len(labels)
        _, pred = torch.max(logits, dim=1)
        accu = torch.mean((pred == labels).float())
        if self.counter == 0:
            self.counter += 1
            self._accu = accu
        else:
            self.counter += 1
            self._accu = self._accu + (accu - self._accu) / float(self.counter)

    def show_state(self):
        if isinstance(self._accu, float):
            print("accuracy: ", self._accu)
        else:
            print("accuracy: ", self._accu.cpu().numpy()[0])

    def get_state(self):
        if isinstance(self._accu, float):
            return self._accu
        else:
            return self._accu.cpu().numpy()[0]

    def reset_state(self):
        self._accu = 0.
        self.counter = 0


class LossMetric(MetricBase):
    def __init__(self):
        super(LossMetric, self).__init__()
        self._loss = 0.
        self.counter = 0

    def update_metric(self, loss):
        if self.counter == 0:
            self.counter += 1
            self._loss = loss
        else:
            self.counter += 1
            self._loss = self._loss + (loss - self._loss) / float(self.counter)

    def show_state(self):
        if isinstance(self._loss, float):
            print("accuracy: ", self._loss)
        else:
            print("accuracy: ", self._loss.cpu().numpy()[0])

    def get_state(self):
        if isinstance(self._loss, float):
            return self._loss
        else:
            return self._loss.cpu().numpy()[0]

    def reset_state(self):
        self._loss = 0.
        self.counter = 0


class IOUMetric(MetricBase):
    def __init__(self):
        super(IOUMetric, self).__init__()
        pass

    def update_metric(self, *inputs, **kwargs):
        pass

    def show_state(self):
        pass

    def get_state(self):
        pass

    def reset_state(self):
        pass


class PrecisionMetric(MetricBase):
    pass


class RecallMetric(MetricBase):
    pass


class MAPMetric(MetricBase):
    pass
