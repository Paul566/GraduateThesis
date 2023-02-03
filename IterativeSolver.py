import warnings

import numpy as np
import typing as tp
from utils import random_spherical_grid, random_spherical_cap_grid, solve_primal, sphere_volume, ball_volume


class IterativeSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 initial_grid_density: int = 100, initial_cap_radius: float = 0.1,
                 density_of_gridpoints_in_unit_cap: float = 100., max_iteration: int = 10,
                 tolerance: float = 1e-9) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.initial_grid_size = int(sphere_volume(dimension) * initial_grid_density)
        self.number_of_gridpoints_in_cap = int(density_of_gridpoints_in_unit_cap * ball_volume(dimension - 1))
        self.cap_radius = initial_cap_radius
        self.x: np.ndarray | None = None
        self.t: float | None = None
        self.max_iteration = max_iteration
        self.tolerance = tolerance

    def extract_initial_based_vectors(self, grid, support_a_values: np.ndarray, support_b_values: np.ndarray):
        ans = []
        for p, supp_a, supp_b in zip(grid, support_a_values, support_b_values):
            if p @ self.x + supp_b * self.t == supp_a:
                ans.append(p)

        if len(ans) > self.dimension:
            return np.array(ans)
        else: # no exact <p, x> + supp(p, B) = supp(p, A), use finite tolerance
            for p, supp_a, supp_b in zip(grid, support_a_values, support_b_values):
                delta = p @ self.x + supp_b * self.t - supp_a
                if delta != 0. and - self.tolerance < delta < self.tolerance:
                    ans.append(p)
            if len(ans) > self.dimension:
                return np.array(ans)
            else:
                raise Exception(f'failed to extract at least n + 1 based vectors')

    def extract_new_based_vectors(self, grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray,
                                  current_based_vectors: np.ndarray) -> np.ndarray:
        num_caps = len(current_based_vectors)
        grid_per_cap = grid.reshape((num_caps, len(grid) // num_caps, self.dimension))
        support_a_values_per_cap = support_a_values.reshape((num_caps, len(grid) // num_caps))
        support_b_values_per_cap = support_b_values.reshape((num_caps, len(grid) // num_caps))
        ans = []
        for grid_in_cap, support_a_values_in_cap, support_b_values_in_cap in \
                zip(grid_per_cap, support_a_values_per_cap, support_b_values_per_cap):
            found_based_vector = False
            for p, supp_a, supp_b in zip(grid_in_cap, support_a_values_in_cap, support_b_values_in_cap):
                if p @ self.x + supp_b * self.t == supp_a:
                    ans.append(p)
                    found_based_vector = True
                    break

            if not found_based_vector: # no exact <p, x> + supp(p, B) = supp(p, A), use finite tolerance
                best_p = grid_in_cap[0]
                best_delta = grid_in_cap[0] @ self.x + support_b_values_in_cap[0] * self.t - support_a_values_in_cap[0]
                for p, supp_a, supp_b in zip(grid_in_cap, support_a_values_in_cap, support_b_values_in_cap):
                    delta = p @ self.x + supp_b * self.t - supp_a
                    if delta < best_delta:
                        best_p = p
                        best_delta = delta

                ans.append(best_p)

                if best_delta > self.tolerance:
                    warnings.warn(f'Failed to find a based vector with tolerance {self.tolerance} in some cap, '
                                  'using the one with minimal (<p, x> + supp(p, B) - supp(p, A))')

        return np.array(ans)

    def solve(self) -> None:
        grid = random_spherical_grid(self.dimension, self.initial_grid_size)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
        based_vectors = self.extract_initial_based_vectors(grid, support_a_values, support_b_values)
        for _ in range(self.max_iteration):
            grid = np.vstack([random_spherical_cap_grid(p, self.cap_radius, self.number_of_gridpoints_in_cap)
                              for p in based_vectors])
            support_a_values = self.support_a(grid)
            support_b_values = self.support_b(grid)
            self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
            based_vectors = self.extract_new_based_vectors(grid, support_a_values, support_b_values, based_vectors)
            self.cap_radius /= 2
