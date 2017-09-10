import visdom
import numpy as np


# TODO: add more Draw... classes


class DrawLine(object):
    def __init__(self, num_line, legend, title):
        """
        using to draw a line
        :param num_line: int, represent the number of line that going to draw.
        :param legend: list of string, represent the meaning of each line.
        :param title: the pane's title
        """
        assert len(legend) == num_line
        self.viz = visdom.Visdom()
        self.window = None
        self.legend = legend
        self.num_line = num_line
        if num_line == 1:
            self.x = np.ones([1]).astype(np.int64)
        else:
            self.x = np.ones([1, self.num_line]).astype(np.int64)
        self.title = title

    def add_point(self, point):
        """
        add a point to the line
        :param point: list or np.ndarray, must have num_line items. e.g. if num_line=2, so point can be
        [1,2], [[1], [2]]
        :return:
        """

        # make sure that point has a valid shape

        # when append, X and Y must have the same shape !!!

        point = self._reshape_point(point)
        if self.window is None:
            # legend setting : once for all
            self.window = self.viz.line(Y=point, X=self.x, opts={'legend': self.legend,
                                                                 'title': self.title})
        else:
            self.viz.line(Y=point, X=self.x, win=self.window, update='append')

        self.x = self.x + 1

    def _reshape_point(self, point):
        assert isinstance(point, (np.ndarray, list))
        self._check_shape(point)
        if isinstance(point, list):
            point = np.array(point)
        point = np.reshape(point, newshape=(1, self.num_line))
        if np.prod(point.shape) == 1:
            point = point.squeeze(axis=0)
        return point

    def _check_shape(self, point):
        if isinstance(point, list):
            point = np.array(point)
        res = np.prod(point.shape)
        if res != self.num_line:
            raise ValueError('the total item in the point must be equal to num_line, excepted item {}'
                             ', but got{}'.format(self.num_line, res))


def main():
    line = DrawLine(num_line=2, legend=['test', 'haha'], title='test')

    line.add_point([1., 2.])
    line.add_point([2., 3.])
    line.add_point([3., 4.])


if __name__ == '__main__':
    main()
