import numpy as np
import itertools
import pyflann

# [reference]: https://github.com/ChangyWen/wolpertinger_ddpg/blob/master/action_space.py
# with modifications

def init_uniform_space(low, high, points):
    dims = len(low)
    # In Discrete situation, the action space is an one dimensional space, i.e., one row
    points_in_each_axis = round(points**(1 / dims))
    # print(points_in_each_axis)

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    # space: e.g., [[1], [2], ... ,[n-1]]
    return np.array(space)

class Space:
    """
    This class represents a n-dimensional unit cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.
    """

    def __init__(self, low, high, points) -> None:
        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)

        self._space_low = -1
        self._space_high = 1

        self._k = (self._space_high - self._space_low) / self._range
        self._space = init_uniform_space([self._space_low] * self._dimensions,
                                          [self._space_high] * self._dimensions,
                                          points)
        
        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def rebuild_flann(self) -> None:
        self._index = self._flann.build_index(self._space, algorithm='kdtree')
    
    def search_point(self, p_in, k):
        if not isinstance(p_in, np.ndarray):
            p_in = np.array([p_in]).astype(np.float64)

        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self._space[search_res]
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]
        
        return knns, np.array(p_out)
        

    def import_point(self, point):
        return self._space_low + self._k * (point - self._low)

    def export_point(self, point):
        """
        map single number to n-d array that match the requirement
        """
        return self._low + (point - self._space_low) / self._k

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]


class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        1, 2, ..., n-1, n
        In gym: 'Discrete' object has no attribute 'high'
    """

    def __init__(self, n):  # n: the number of the discrete actions
        super().__init__([1], [n], n)

    def export_point(self, point):
        return np.round(super().export_point(point)).astype(int)


class Binary_space(Space):
    """
    TODO: Complete this later for the model-free control paper
    """
    pass


if __name__ == '__main__':
    """ Tests """
    # print(init_uniform_space(np.array([0,0,0]), np.array([1,1,1]), 1000))
    ds = Space([-2], [2], 200000)
    output, output_2 = ds.search_point(1.4, 5)
    print(output_2)
    print(output_2.shape)

