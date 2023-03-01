import numpy as np
import typing as tp
from utils import random_spherical_grid, spherical_cap_crosslike_grid, solve_primal
from sklearn.cluster import KMeans


class GreedySolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 num_iterations: int = 1000, final_cross_radius = 1e-8, tolerance = 1e-12,
                 grid_size_max_inflation: int = 2, max_number_restarts: int = 30) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.t: float | None = None
        self.x: np.ndarray | None = None
        self.best_t: float = - np.inf
        self.best_x: np.ndarray | None = None
        self.num_iterations = num_iterations
        self.final_cross_radius = final_cross_radius
        self.tolerance = tolerance
        self.grid_size_max_inflation = grid_size_max_inflation
        self.current_try = 0

    def extract_new_based_vectors(self, grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray,
                                  current_based_vectors: np.ndarray) -> np.ndarray:
        num_caps = len(current_based_vectors)
        grid_per_cap = grid.reshape((num_caps, 2 * self.dimension - 1, self.dimension))
        support_a_values_per_cap = support_a_values.reshape((num_caps, 2 * self.dimension - 1))
        support_b_values_per_cap = support_b_values.reshape((num_caps, 2 * self.dimension - 1))
        ans = []
        for grid_in_cap, support_a_values_in_cap, support_b_values_in_cap in \
                zip(grid_per_cap, support_a_values_per_cap, support_b_values_per_cap):
            differences = grid_in_cap @ self.x + self.t * support_b_values_in_cap - support_a_values_in_cap
            index = np.argmin(differences)
            ans.append(grid_in_cap[index])

        return np.array(ans)

    def extract_initial_based_vectors(self, grid: np.ndarray, support_a_values: np.ndarray,
                                      support_b_values: np.ndarray) -> np.ndarray:
        """
        :param grid: current grid
        :param support_a_values:
        :param support_b_values:
        :return: returns (dimension * (dimension + 1)) grid elements with least (<p, x> + supp(p, B) - supp(p, A))
        """
        differences = grid @ self.x + self.t * support_b_values - support_a_values
        size_of_output = (self.dimension + 1) * self.dimension
        indices = np.argpartition(differences, size_of_output)[:size_of_output]
        return grid[indices]

    def extract_based_vectors_with_tolerance(self, grid: np.ndarray, support_a_values: np.ndarray,
                                      support_b_values: np.ndarray) -> np.ndarray:
        """
        :param grid:
        :param support_a_values:
        :param support_b_values:
        :return: grid elements such that
        """
        differences = grid @ self.x + self.t * support_b_values - support_a_values
        indices = np.where(differences < self.tolerance)[0]
        return grid[indices]

    def next_based_vectors_heuristic(self, grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray,
                                  current_based_vectors: np.ndarray, cross_radius: float) -> np.ndarray:
        candidate_vectors = np.unique(
                np.vstack(
                (self.extract_new_based_vectors(grid, support_a_values, support_b_values, current_based_vectors),
                self.extract_based_vectors_with_tolerance(grid, support_a_values, support_b_values))
                ), axis=0)
        return candidate_vectors

    def inner_solve(self) -> None:
        initial_grid_size = (self.dimension * 2 - 1) * (self.dimension + 1) * self.dimension
        grid = random_spherical_grid(self.dimension, initial_grid_size)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
        based_vectors = self.extract_initial_based_vectors(grid, support_a_values, support_b_values)
        inflatable_grid = True
        for iteration in range(self.num_iterations):
            cross_radius = np.pi / 2 * np.power(2 / np.pi * self.final_cross_radius, iteration / self.num_iterations)

            grid = np.vstack([spherical_cap_crosslike_grid(p, cross_radius) for p in based_vectors])
            support_a_values = self.support_a(grid)
            support_b_values = self.support_b(grid)

            # if iteration % 50 == 0:
            #     print(f'iteration {iteration} \t |x| = {np.linalg.norm(self.x)} \t cross_radius = {cross_radius} \t t = {self.t}')

            try:
                self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
                if self.t > self.best_t:
                    self.best_t, self.best_x = self.t, self.x
            except TypeError: # if the linpog problem is unbounded, start over, unless we are out of tries
                self.current_try += 1
                if self.current_try == self.num_iterations:
                    return
                self.inner_solve()
                return

            if inflatable_grid:
                based_vectors = self.next_based_vectors_heuristic(grid, support_a_values, support_b_values, based_vectors, cross_radius)
            else:
                based_vectors = self.extract_new_based_vectors(grid, support_a_values, support_b_values, based_vectors)

            if len(grid) > initial_grid_size * self.grid_size_max_inflation:
                kmeans = KMeans(n_clusters=self.dimension + 1).fit(based_vectors)
                based_vectors = kmeans.cluster_centers_
                inflatable_grid = False

    def solve(self) -> None:
        self.inner_solve()
        #best_x, best_t = self.x, self.t
        #for _ in range(self.number_restarts - 1):
        #    self.inner_solve()
        #    if self.t - self.tolerance < best_t < self.t + self.tolerance: # we probably hit the optimum two times
        #        break
        self.x, self.t = self.best_x, self.best_t
