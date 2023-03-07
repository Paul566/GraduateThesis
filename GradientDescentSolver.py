import numpy as np
import typing as tp
from scipy.stats import special_ortho_group
from utils import solve_primal


class GradientDescentSolver:
    def __init__(self, dimension: int, support_with_argmax_a: tp.Callable, support_with_argmax_b: tp.Callable,
                 num_iterations: int = 1000, learning_rate = 0.1, initial_regularization_coefficient = 0.1,
                 tolerance = 1e-12, max_num_restarts = 1000) -> None:
        self.support_with_argmax_a = support_with_argmax_a
        self.support_with_argmax_b = support_with_argmax_b
        # support_with_argmax should return a tuple of arrays: (support values and argmaxes)
        self.dimension = dimension
        self.t: float | None = None
        self.x: np.ndarray | None = None
        self.grid: np.ndarray | None = None
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.initial_regularization_coefficient = initial_regularization_coefficient
        self.tolerance = tolerance
        self.restarts = 0
        self.max_num_restarts = max_num_restarts

    def zero_inside_convex_hull(self, simplex):
        new_basis = simplex[1:] - np.tile(simplex[0, :], (self.dimension, 1))
        try:
            zero_in_new_basis = - simplex[0, :] @ np.linalg.inv(new_basis)
        except np.linalg.LinAlgError:  # degenerate case
            return False
        return np.all(zero_in_new_basis > 0) and np.sum(zero_in_new_basis) < 1

    def feasibilifying_multipliers(self, simplex) -> np.ndarray:
        """
        one can make grid feasible by changing some gridpoints to their antipodes
        :return: array of +-1
        """
        new_basis = simplex[1:] - np.tile(simplex[0, :], (self.dimension, 1))
        try:
            zero_in_new_basis = - simplex[0, :] @ np.linalg.inv(new_basis)
        except np.linalg.LinAlgError:  # degenerate case
            return np.ones(self.dimension + 1)

        return np.hstack((np.array(float(np.sum(zero_in_new_basis) <= 1) * 2 - 1),
                                 np.sign(zero_in_new_basis)))

    def get_t_gradient_for_simplex(self, simplex, support_a_values, support_a_argmaxes,
                               support_b_values, support_b_argmaxes) -> np.ndarray:
        M_a = np.hstack((support_a_values.reshape((self.dimension + 1, 1)), simplex))
        M_b = np.hstack((support_b_values.reshape((self.dimension + 1, 1)), simplex))

        M_a_inv = np.linalg.inv(M_a)
        M_b_inv = np.linalg.inv(M_b)

        det_M_a = np.linalg.det(M_a)
        det_M_b = np.linalg.det(M_b)

        det_M_a_gradient = det_M_a * (M_a_inv[1:].T + support_a_argmaxes *
                                      np.tile(M_a_inv[0].reshape((self.dimension + 1, 1)), (1, self.dimension)))
        det_M_b_gradient = det_M_b * (M_b_inv[1:].T + support_b_argmaxes *
                                      np.tile(M_b_inv[0].reshape((self.dimension + 1, 1)), (1, self.dimension)))

        t_gradient = (det_M_a_gradient * det_M_b - det_M_b_gradient * det_M_a) / (det_M_b ** 2)
        return t_gradient

    def get_based_gridpoints_indices(self, support_a_values, support_b_values) -> np.ndarray:
        differences = self.grid @ self.x + self.t * support_b_values - support_a_values
        size_of_output = self.dimension + 1
        return np.argpartition(differences, size_of_output)[:size_of_output]

    def update_grid_t_x(self):
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)

        self.t, self.x = solve_primal(self.grid, support_a_values, support_b_values)

        based_gridpoints_indices = self.get_based_gridpoints_indices(support_a_values, support_b_values)

        print(f'based_gridpoint_indices = {based_gridpoints_indices}')

        simplex = self.grid[based_gridpoints_indices]
        simplex_support_a_values = support_a_values[based_gridpoints_indices]
        simplex_support_b_values = support_b_values[based_gridpoints_indices]
        simplex_support_a_argmaxes = support_a_argmaxes[based_gridpoints_indices]
        simplex_support_b_argmaxes = support_b_argmaxes[based_gridpoints_indices]
        t_gradient_for_simplex = self.get_t_gradient_for_simplex(simplex, simplex_support_a_values,
                                simplex_support_a_argmaxes, simplex_support_b_values, simplex_support_b_argmaxes)

        print(f'gradient: \n {t_gradient_for_simplex}')

        for i, index in enumerate(based_gridpoints_indices):
            self.grid[index] += self.learning_rate * t_gradient_for_simplex[i]
            self.grid[index - self.dimension - 1] -= self.learning_rate * t_gradient_for_simplex[i]

        print(f'uncorrected grid: \n {self.grid}')

        self.grid = (self.grid.T / np.linalg.norm(self.grid, axis=1)).T
        multipliers = self.feasibilifying_multipliers(self.grid[:self.dimension + 1])
        self.grid = self.grid * np.tile(multipliers.reshape(self.dimension + 1, 1), (2, self.dimension))

    def get_t_and_x(self) -> tp.Tuple[float, np.ndarray]:
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
        M_b = np.hstack((support_b_values.reshape((self.dimension + 1, 1)), self.grid))
        t_and_x_vector = np.linalg.inv(M_b) @ support_a_values
        return t_and_x_vector[0], t_and_x_vector[1:]

    def regular_simplex(self) -> np.ndarray:
        n = self.dimension
        first_n_vertices = np.sqrt(1 + 1. / n) * np.identity(n) - \
                           np.power(n, -3. / 2) * (np.sqrt(n + 1) - 1) * np.ones((n, n))
        last_vertex = - 1. / np.sqrt(n) * np.ones(n)
        return np.vstack((first_n_vertices, last_vertex))

    def solve(self):
        # TODO estimate the good learning_rate
        regular_simplex = self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)
        self.grid = np.vstack((regular_simplex, - regular_simplex))
        # grid always consists of dim+1 rows with vertices and dim+1 rows with their antipodes

        self.t = - np.inf
        delta_t = np.inf
        iteration = 0
        while iteration < self.num_iterations and np.abs(delta_t) > self.tolerance:
            iteration += 1
            previous_t = self.t
            self.update_grid_t_x()
            delta_t = self.t - previous_t

            print(f'iteration {iteration} \t t {self.t}')
            print(self.grid)
