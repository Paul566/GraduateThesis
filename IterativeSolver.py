import sys
import numpy as np
import typing as tp
from utils import sphere_grid_from_cube, spherical_cap_grid_from_cube, solve_primal, sphere_volume, ball_volume


class IterativeSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 initial_grid_density: int = 100, max_iteration: int = 50, tolerance: float = 1e-12) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.initial_grid_size = int(sphere_volume(dimension) * initial_grid_density)
        self.number_of_gridpoints_in_cap = int(self.initial_grid_size / (dimension + 1))
        self.cap_radius = np.pi / 2
        self.x: np.ndarray | None = None
        self.t: float | None = None
        self.best_t: np.ndarray | None = None
        self.best_x: np.ndarray | None = None
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.iteration = 0

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
            best_p = grid_in_cap[0]
            best_delta = grid_in_cap[0] @ self.x + support_b_values_in_cap[0] * self.t - support_a_values_in_cap[0]
            for p, supp_a, supp_b in zip(grid_in_cap, support_a_values_in_cap, support_b_values_in_cap):
                delta = p @ self.x + supp_b * self.t - supp_a
                if delta < best_delta:
                    best_p = p
                    best_delta = delta

            ans.append(best_p)

            # if best_delta > self.tolerance:
            #     sys.stderr.write(f'Warning: failed to find a based vector with tolerance {self.tolerance} in some cap, '
            #                   'using the one with minimal (<p, x> + supp(p, B) - supp(p, A))\n')

        return np.array(ans)

    def solve(self) -> None:
        grid = sphere_grid_from_cube(self.dimension, self.initial_grid_size)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
        based_vectors = self.extract_initial_based_vectors(grid, support_a_values, support_b_values)
        while self.iteration < self.max_iteration:
            self.iteration += 1
            grid = np.vstack([spherical_cap_grid_from_cube(p, self.cap_radius, self.number_of_gridpoints_in_cap)
                              for p in based_vectors])
            support_a_values = self.support_a(grid)
            support_b_values = self.support_b(grid)
            try:
                self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
            except TypeError:
                # if the linprog problem is unbounded, we lost some based vectors from the search space,
                # let's start over with finer grids
                self.cap_radius = np.pi / 2
                self.number_of_gridpoints_in_cap *= 2
                self.initial_grid_size *= 2
                self.solve()
                return

            based_vectors = self.extract_new_based_vectors(grid, support_a_values, support_b_values, based_vectors)
            self.cap_radius /= 2


