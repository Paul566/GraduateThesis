import numpy as np
import typing as tp
from utils import random_spherical_grid, spherical_cap_crosslike_grid, solve_primal


class GreedySolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 num_iterations: int = 1000, final_cross_radius = 1e-8, tolerance = 1e-12) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.x: np.ndarray | None = None
        self.t: float | None = None
        self.num_iterations = num_iterations
        self.final_cross_radius = final_cross_radius

    def extract_new_based_vectors(self, grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray,
                                  current_based_vectors: np.ndarray) -> np.ndarray:
        num_caps = len(current_based_vectors)
        grid_per_cap = grid[:- 2 * self.dimension].reshape((num_caps, 2 * self.dimension - 1, self.dimension))
        support_a_values_per_cap = support_a_values[:- 2 * self.dimension].reshape((num_caps, 2 * self.dimension - 1))
        support_b_values_per_cap = support_b_values[:- 2 * self.dimension].reshape((num_caps, 2 * self.dimension - 1))
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
        :return: returns (dimension * (dimension + 1)) grid elements with least (<p, x> + supp(p, B) = supp(p, A))
        """
        differences = grid @ self.x + self.t * support_b_values - support_a_values
        size_of_output = (self.dimension + 1) * self.dimension
        indices = np.argpartition(differences, size_of_output)[:size_of_output]
        return grid[indices]

    def solve(self) -> None:
        initial_grid_size = (self.dimension * 2 - 1) * (self.dimension + 1) * self.dimension
        grid = random_spherical_grid(self.dimension, initial_grid_size)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
        based_vectors = self.extract_initial_based_vectors(grid, support_a_values, support_b_values)
        for iteration in range(self.num_iterations):
            cross_radius = np.pi / 2 * np.power(2 / np.pi * self.final_cross_radius, iteration / self.num_iterations)

            # if iteration % 10 == 0:
            #     print(f'iteration {iteration} \t |x| = {np.linalg.norm(self.x)} \t cross_radius = {cross_radius} \t t = {self.t}')

            # add +-basis vectors to the grid to make sure the problem is bounded
            grid = np.vstack([spherical_cap_crosslike_grid(p, cross_radius) for p in based_vectors] +
                             [np.identity(self.dimension), - np.identity(self.dimension)])
            support_a_values = self.support_a(grid)
            support_b_values = self.support_b(grid)
            self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
            based_vectors = self.extract_new_based_vectors(grid, support_a_values, support_b_values, based_vectors)
