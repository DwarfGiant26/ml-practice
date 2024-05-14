import numpy as np
from helper_class import *

"""
K-means clustering:

- pick k different starting points for clustering
- For each point, compute the cluster closest to it
- Recompute the centroid of each cluster
- Repeat until the centroid does not change or until the maximum number of iteration is reached
"""

class KMeans():
    @staticmethod
    def get_closest_centroid_index(point: np.array, centroids:np.array) -> np.array:
        return np.argmin(np.array([Vector2.get_euclidean_distance(point, centroid) for centroid in centroids]))

    @staticmethod
    def pick_k_random_points(points: np.array, k: int):
        num_points = points.shape[0]
        choices = np.random.choice(num_points, k, replace=False)
        return points[choices]

    @staticmethod
    def k_means(points: np.array, k: int, max_iters: int = 100) -> list[Cluster]:
        centroids = KMeans.pick_k_random_points(points, k)

        for _ in range(max_iters):
            clusters: list[Cluster] = [Cluster() for i in range(k)]
            assert(len(centroids) == k)
            # calculate closest centroid/cluster
            for point in points:
                closest_centroid_index = KMeans.get_closest_centroid_index(point, centroids)
                clusters[closest_centroid_index].add_point(point)

            # recalculate centroid
            centroids = [cluster.centroid() for cluster in clusters]

        return clusters
