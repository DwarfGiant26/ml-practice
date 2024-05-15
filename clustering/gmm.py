"""
Gaussian Mixture Model is a clustering algorithm that assumes that the data points follows normal distribution.
Basically each of the cluster is a single distribution with mean and standard deviation being parameterized.

Steps:
- Estimate the probability of each point belonging to each distribution
- Recompute the mean and variance of the distribution
- Repeat until the mean and variance does not change or maximum number of iterations has been reached.
"""

from helper_class import *
import math

np.random.seed(0)


class NormalDistribution:
    def __init__(self, mean: float, std: float):
        self.mean: float = mean
        self.std: float = std

    def probability_density_at(self, value: float) -> float:
        return (1 / (self.std * np.sqrt(2 * math.pi))) * math.exp(-(value - self.mean)**2 / (2* self.std**2))

class GMM:
    def __init__(self, num_clusters: int):
        self.num_clusters:int = num_clusters
        self.distributions:list[NormalDistribution] = [NormalDistribution(np.random.uniform(0, 1.0), np.random.uniform(0, 1.0)) for i in range(num_clusters)]

    """
        Return the probability of a point being in each distribution. 
        The sum of the probabilities has to add up to 1.  
    """
    def estimate_probabilities(self, point:np.array) -> np.array:
        non_normalized_probabilities = [np.array([distribution.probability_density_at(point[0]), distribution.probability_density_at(point[1])]) for distribution in self.distributions]
        total = np.sum(non_normalized_probabilities, axis=0)
        probabilities = np.array([p/total for p in non_normalized_probabilities])

        return probabilities

    def recompute_distribution(self):
        pass

    def get_clusters(self, points: np.ndarray, max_iters: int, diff_threshold: float = 1e-6) -> list[Cluster]:
        for _ in range(max_iters):
            # ps is a matrix of probabilities with the row being which points and col being which cluster
            ps = np.array([self.estimate_probabilities(point) for point in points])
            total_p_each_cluster = np.sum(ps, axis=0)
            normalized_ps = ps / total_p_each_cluster # TODO(Probably need to copy the rows?)

            # Matrix of new parameters for distribution with format [[meanx1, meanx2, ...], [meany1, meany2, ...], ...]
            new_params = points @ normalized_ps

            # TODO: Check for diff smaller than epsilon case

            # update distribution
            self.distributions = [NormalDistribution(new_params[0, i], new_params[0, i]) for i in range(self.num_clusters)]

        best_clusters = np.argmax(normalized_ps, axis=1)
        clusters = [Cluster() for _ in range(self.num_clusters)]
        for i in range(len(points)):
            clusters[best_clusters[i]] = points[i]

        return clusters



