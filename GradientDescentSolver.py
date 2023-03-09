import numpy as np
import typing as tp
from scipy.stats import special_ortho_group
from utils import solve_primal


class GradientDescentSolver:
    def __init__(self, dimension: int, support_with_argmax_a: tp.Callable, support_with_argmax_b: tp.Callable,
                 max_num_iterations: int = 1000, learning_rate = 0.1,
                 tolerance = 1e-8) -> None:
        """
        :param dimension:
        :param support_with_argmax_a: should return a tuple of arrays: (support values and argmaxes)
        :param support_with_argmax_b:
        :param max_num_iterations:
        :param learning_rate:
        :param tolerance:
        """
        self.support_with_argmax_a = support_with_argmax_a
        self.support_with_argmax_b = support_with_argmax_b
        self.dimension = dimension
        self.t: float | None = None
        self.x: np.ndarray | None = None
        self.grid: np.ndarray | None = None
        self.max_num_iterations = max_num_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.iteration = 0

    def zero_inside_convex_hull(self, simplex):
        new_basis = simplex[1:] - np.tile(simplex[0, :], (self.dimension, 1))
        try:
            zero_in_new_basis = - simplex[0, :] @ np.linalg.inv(new_basis)
        except np.linalg.LinAlgError:  # degenerate case
            return False
        return np.all(zero_in_new_basis > 0) and np.sum(zero_in_new_basis) < 1

    def get_based_gridpoints_indices(self, support_a_values, support_b_values) -> np.ndarray:
        differences = self.grid @ self.x + self.t * support_b_values - support_a_values
        size_of_output = self.dimension + 1
        return np.argpartition(differences, size_of_output)[:size_of_output]

    def get_t_and_x(self, support_a_values, support_b_values) -> tp.Tuple[float, np.ndarray]:
        M_b = np.hstack((support_b_values.reshape((self.dimension + 1, 1)), self.grid))
        t_and_x_vector = np.linalg.inv(M_b) @ support_a_values
        return t_and_x_vector[0], t_and_x_vector[1:]

    def regular_simplex(self) -> np.ndarray:
        n = self.dimension
        first_n_vertices = np.sqrt(1 + 1. / n) * np.identity(n) - \
                           np.power(n, -3. / 2) * (np.sqrt(n + 1) - 1) * np.ones((n, n))
        last_vertex = - 1. / np.sqrt(n) * np.ones(n)
        return np.vstack((first_n_vertices, last_vertex))

    def update_grid_t_x(self) -> None:
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
        grid_gradient = np.tile(self.x, (len(self.grid), 1)) + self.t * support_b_argmaxes - support_a_argmaxes
        self.grid -= self.learning_rate * grid_gradient
        self.grid = (self.grid.T / np.linalg.norm(self.grid, axis=1)).T

        #print(f'grid (mb not feasible) \n {self.grid}')

        try:
            support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
            support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
            self.t, self.x = solve_primal(self.grid, support_a_values, support_b_values)
        except TypeError:
            # in case the relaxed grid doesn't contain zero in its convex hull
            self.grid = np.vstack((self.grid,
                                   self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)))
            support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
            support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
            self.t, self.x = solve_primal(self.grid, support_a_values, support_b_values)
            #self.grid = self.grid[self.get_based_gridpoints_indices(support_a_values, support_b_values)]

    def solve(self):
        self.grid = self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
        self.t, self.x = self.get_t_and_x(support_a_values, support_b_values)

        delta_t = np.inf
        while self.iteration < self.max_num_iterations and np.abs(delta_t) > self.tolerance:
            self.iteration += 1
            previous_t = self.t
            self.update_grid_t_x()
            delta_t = self.t - previous_t

            #print(f'iteration {self.iteration}\t t {self.t}\t x {self.x}')
