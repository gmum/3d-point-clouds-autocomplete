import os
from os.path import join

import numpy as np
import ray

from utils.plyfile import load_ply, quick_save_ply_file


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

    def __init__(self, root_dir=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

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

    @ray.remote
    def __generating_slices(self, category: str, filename: str):
        pc_filepath = join(self.root_dir, category, filename)
        points = load_ply(pc_filepath)

        if self.transform:
            points = self.transform(points)

        for i in range(4):
            real, remaining = self.generate_item(points)

            quick_save_ply_file(real, self.root_dir + '/slices/real/' +
                                category + '/' + str(i) + '~' + filename)
            quick_save_ply_file(remaining, self.root_dir + '/slices/remaining/' +
                                category + '/' + str(i) + '~' + filename)

    def generate(self, pc_df_iter):
        ray.init(num_cpus=os.cpu_count())
        ray.get([self.__generating_slices.remote(self, row['category'], row['filename']) for _, row in
                 pc_df_iter])
        ray.shutdown()
