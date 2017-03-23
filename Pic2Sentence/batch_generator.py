class BatchGenerator(object):
    def __init__(self, inputs, targets, batch_size):
        self._inputs = inputs
        self._targets = targets
        self._batch_size = batch_size
        self._counter = 0
    @property
    def inputs(self):
        return self._inputs
    @property
    def targets(self):
        return self._targets
    @property
    def batch_size(self):
        return self._batch_size
    def next_batch(self):
        pass
