import numpy as np
import typing as tp
from utils import sphere_grid_from_cube_with_random_rotation, spherical_cap_grid_from_cube, solve_primal


class IterativeSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 cap_grid_diameter: int = 3, tolerance: float = 1e-12, number_restarts: int = 10) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.cap_grid_diameter = cap_grid_diameter
        self.x: np.ndarray | None = None
        self.t: float | None = None
        self.best_t: float | None = None
        self.best_x: np.ndarray | None = None
        self.tolerance = tolerance
        self.number_restarts = number_restarts

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
        :return: returns (dimension * (dimension + 1)) grid elements with least (<p, x> + supp(p, B) - supp(p, A))
        """
        differences = grid @ self.x + self.t * support_b_values - support_a_values
        size_of_output = self.dimension * (self.dimension + 1)
        indices = np.argpartition(differences, size_of_output)[:size_of_output]
        return grid[indices]

    def inner_solve(self) -> None:
        cap_radius = np.pi / 2
        grid = sphere_grid_from_cube_with_random_rotation(self.dimension, self.cap_grid_diameter * (self.dimension + 1) // 2)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
        based_vectors = self.extract_initial_based_vectors(grid, support_a_values, support_b_values)
        while cap_radius > self.tolerance:
            grid = np.vstack([spherical_cap_grid_from_cube(p, cap_radius, self.cap_grid_diameter)
                              for p in based_vectors])
            support_a_values = self.support_a(grid)
            support_b_values = self.support_b(grid)
            try:
                self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
            except TypeError: # if the linprog problem is unbounded, start over
                return

            based_vectors = self.extract_new_based_vectors(grid, support_a_values, support_b_values, based_vectors)
            cap_radius /= ((1 + np.sqrt(5)) / 2) # make cap_radius phi times smaller

    def solve(self) -> None:
        self.inner_solve()
        self.best_x, self.best_t = self.x, self.t
        for _ in range(self.number_restarts - 1):
            self.inner_solve()
            if self.t - self.tolerance < self.best_t < self.t + self.tolerance: # we probably hit the optimum two times
                break
            if self.t > self.best_t:
                self.best_x, self.best_t = self.x, self.t
        self.x, self.t = self.best_x, self.best_t
