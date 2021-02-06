import numpy as np


class HyperPlane(object):

    def __init__(self, params, bias):
        self.params = params
        self.bias = bias

    def check_point(self, point):
        return np.sign(np.dot(point, self.params) + self.bias)

    @staticmethod
    def get_plane_from_3_points(points):
        cp = np.cross(points[1] - points[0], points[2] - points[0])
        return HyperPlane(cp, np.dot(cp, points[0]))

    @staticmethod
    def get_random_plane():
        return HyperPlane.get_plane_from_3_points(np.random.rand(3, 3))

    def __str__(self):
        return "Plane A={}, B={}, C={}, D={}".format(*self.params, self.bias)


class SlicedDatasetGenerator(object):

    @staticmethod
    def generate_item(points, target_partition_points=1024):

        while True:
            under = HyperPlane.get_random_plane().check_point(points) > 0
            points_under_plane = points[under]
            points_above_plane = points[~under]

            if target_partition_points == len(points_under_plane):
                return points_under_plane, points_above_plane
            if target_partition_points == len(points_above_plane):
                return points_above_plane, points_under_plane
