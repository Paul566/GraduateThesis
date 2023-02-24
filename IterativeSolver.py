import sys
import numpy as np
import typing as tp
from utils import sphere_grid_from_cube, spherical_cap_grid_from_cube, solve_primal, sphere_volume, ball_volume


class IterativeSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 cap_grid_diameter: int = 3, max_iteration: int = 50, tolerance: float = 1e-12) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.cap_grid_diameter = cap_grid_diameter
        self.cap_radius = np.pi / 2
        self.x: np.ndarray | None = None
        self.t: float | None = None
        self.best_t: np.ndarray | None = None
        self.best_x: np.ndarray | None = None
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.iteration = 0

    def extract_new_based_vectors(self, grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray,
                                  current_based_vectors: np.ndarray) -> np.ndarray:
        num_caps = len(current_based_vectors)
        grid_per_cap = grid.reshape((num_caps, len(grid) // num_caps, self.dimension))
        support_a_values_per_cap = support_a_values.reshape((num_caps, len(grid) // num_caps))
        support_b_values_per_cap = support_b_values.reshape((num_caps, len(grid) // num_caps))
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
        size_of_output = self.dimension * (self.dimension + 1)
        indices = np.argpartition(differences, size_of_output)[:size_of_output]
        return grid[indices]

    def solve(self) -> None:
        grid = sphere_grid_from_cube(self.dimension, self.cap_grid_diameter * (self.dimension + 1) // 2)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
        based_vectors = self.extract_initial_based_vectors(grid, support_a_values, support_b_values)
        while self.iteration < self.max_iteration:
            self.iteration += 1
            grid = np.vstack([spherical_cap_grid_from_cube(p, self.cap_radius, self.cap_grid_diameter)
                              for p in based_vectors])
            support_a_values = self.support_a(grid)
            support_b_values = self.support_b(grid)
            try:
                self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
            except TypeError:
                # if the linprog problem is unbounded, we lost some based vectors from the search space,
                # let's start over with finer grids
                self.cap_radius = np.pi / 2
                self.cap_grid_diameter *= 2
                self.solve()
                return

            based_vectors = self.extract_new_based_vectors(grid, support_a_values, support_b_values, based_vectors)
            self.cap_radius /= ((1 + np.sqrt(5)) / 2) # make cap_radius phi times smaller
