from datasets.shapenet import ShapeNetDataset
import numpy as np
import pptk


def show_3d_cloud(points_cloud):
    pptk.viewer(points_cloud).set(show_axis=False)


def quick_save_ply_file(points, filename):
    pl = len(points)
    header = \
        "ply\n" \
        "format binary_little_endian 1.0\n" \
        "element vertex " + str(pl) + "\n" \
        "property float x\n" \
        "property float y\n" \
        "property float z\n" \
        "end_header\n"

    dtype_vertex = [('vertex', '<f4', (3))]
    vertex = np.empty(pl, dtype=dtype_vertex)
    vertex['vertex'] = points

    with open(filename, 'wb') as fp:
        fp.write(bytes(header, encoding='utf-8'))
        fp.write(vertex.tobytes())
        fp.close()


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


def generate_item(points_cloud, min_partition=0.4):
    points = points_cloud[0]
    current_partition = 0

    points_under_plane = None
    points_above_plane = None

    while current_partition < min_partition:
        under = HyperPlane.get_random_plane().check_point(points) > 0
        points_under_plane = points[under]
        points_above_plane = points[under == False]
        part = len(points_under_plane) / len(points)
        current_partition = np.min([part, 1. - part])

    show_3d_cloud(points)
    show_3d_cloud(points_under_plane)
    show_3d_cloud(points_above_plane)

    # quick_save_ply_file(points, "full.ply")
    # quick_save_ply_file(points_above_plane, "above.ply")


if __name__ == '__main__':
    shape_net_dataset = ShapeNetDataset(root_dir="/3d_point_cloud/dataset/shapenet")

    # show_3d_cloud(load_ply("above.ply"))

    generate_item(shape_net_dataset[0])
