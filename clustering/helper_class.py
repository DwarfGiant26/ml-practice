import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

class Vector2:
    def __init__(self, x, y) -> None:
        self.vec = np.array([x,y])

    @staticmethod
    def get_euclidean_distance(a: np.array, b: np.array) -> float:
        assert(a.shape == b.shape)
        return np.sqrt(np.sum((a - b)**2))
class Cluster:
    def __init__(self, points: np.array = np.empty((0,2))) -> None:
        if points.shape[0] > 0:
            assert(points.ndim == 2)
            assert(points.shape[1] == 2)
        self.points = points

    def add_point(self, point: np.array):
        self.points = np.append(self.points, np.array([point]), axis=0)

    def centroid(self) -> Vector2:
        return np.average(self.points, axis=0)

class ClusteringVizHelper:
    @staticmethod
    def plot_clusters(clusters: list[Cluster]) -> None:
        xs = []
        ys = []
        colors = []
        for color_i, cluster in enumerate(clusters):
            xs += cluster.points[:, 0].tolist()
            ys += cluster.points[:, 1].tolist()
            colors += [color_i for _ in range(cluster.points.shape[0])]
        plt.scatter(xs, ys, c=colors)

class ClusteringTestGenerator:
    @staticmethod
    def generate_points(num_clusters: int, num_points: int = 100) -> np.array:
        points, y_true = make_blobs(n_samples=num_points, centers=num_clusters, cluster_std=0.60, random_state=0)
        return np.array(points)

    @staticmethod
    def generate_streched_points(num_clusters: int, num_points: int = 100) -> np.array:
        rng = np.random.RandomState(74)
        # transform the data to be stretched
        transformation = rng.normal(size=(2, 2))
        points = ClusteringTestGenerator.generate_points(num_clusters, num_points)
        points = np.dot(points, transformation)

        return points

    def generate_moon(num_points: int = 100, noise: float = .05) -> np.array:
        return make_moons(200, noise=noise, random_state=0)

if __name__ == "__main__":
    points = ClusteringTestGenerator.generate_points(num_clusters=5)

