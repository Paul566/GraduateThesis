import numpy as np
import typing as tp
from scipy.stats import special_ortho_group


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

    def zero_inside_convex_hull(self):
        new_basis = self.grid[1:] - np.tile(self.grid[0, :], (self.dimension, 1))
        try:
            zero_in_new_basis = - self.grid[0, :] @ np.linalg.inv(new_basis)
        except np.linalg.LinAlgError:  # degenerate case
            return False
        return np.all(zero_in_new_basis > 0) and np.sum(zero_in_new_basis) < 1

    def feasiblify_grid(self) -> bool:
        """
        makes grid feasible by changing some gridpoints to their antipodes
        :return: true if the grid was feasible, else false
        """
        new_basis = self.grid[1:] - np.tile(self.grid[0, :], (self.dimension, 1))
        try:
            zero_in_new_basis = - self.grid[0, :] @ np.linalg.inv(new_basis)
        except np.linalg.LinAlgError:  # degenerate case
            return

        multipliers = np.hstack((np.array(float(np.sum(zero_in_new_basis) <= 1) * 2 - 1),
                                 np.sign(zero_in_new_basis)))
        self.grid = self.grid * np.tile(multipliers.reshape(self.dimension + 1, 1), (1, self.dimension))

        return np.all(multipliers > 0)


    def t_gradient(self) -> np.ndarray:
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)

        M_a = np.hstack((support_a_values.reshape((self.dimension + 1, 1)), self.grid))
        M_b = np.hstack((support_b_values.reshape((self.dimension + 1, 1)), self.grid))

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

    def regularization_gradient(self) -> np.ndarray:
        ans = np.zeros((self.dimension + 1, self.dimension))
        indices = np.arange(self.dimension + 1)

        # TODO avoid loops
        for k in range(self.dimension + 1):
            plane = self.grid[indices != k, :]
            matrix = np.vstack((plane[1:] - np.tile(plane[0, :], (self.dimension - 1, 1)), np.ones(self.dimension)))
            normal = np.linalg.inv(matrix) @ np.hstack((np.zeros(self.dimension - 1), np.array([1])))
            normal = normal / np.linalg.norm(normal)
            signed_delta = plane[0] @ normal

            for i in range(self.dimension + 1):
                if i != k:
                    ans[i] += normal / signed_delta * np.sqrt(np.abs(signed_delta))

        return ans * self.initial_regularization_coefficient

    def delta(self) -> np.ndarray:
        ans = np.inf
        indices = np.arange(self.dimension + 1)

        for k in range(self.dimension + 1):
            plane = self.grid[indices != k, :]
            matrix = np.vstack((plane[1:] - np.tile(plane[0, :], (self.dimension - 1, 1)), np.ones(self.dimension)))
            normal = np.linalg.inv(matrix) @ np.hstack((np.zeros(self.dimension - 1), np.array([1])))
            normal = normal / np.linalg.norm(normal)
            delta = np.abs(plane[0] @ normal)
            if delta < ans:
                ans = delta

        return ans

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

    def inner_solve(self) -> bool:
        """
        :return: success / fail
        """
        # TODO estimate the good learning_rate
        self.grid = self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)
        self.t, self.x = self.get_t_and_x()
        delta_t = np.inf
        iteration = 0
        last_change = False
        while iteration < self.num_iterations and np.abs(delta_t) > self.tolerance:
            iteration += 1
            gradient = self.t_gradient()

            #if i < self.num_iterations:
            #    gradient += self.regularization_gradient() * (self.num_iterations - i) / self.num_iterations

            self.grid += self.learning_rate * gradient
            #if not self.zero_inside_convex_hull():
            #    return False
            last_change = self.feasiblify_grid() # last_change = (was the grid already feasible?)
            self.grid = (self.grid.T / np.linalg.norm(self.grid, axis=1)).T

            previous_t = self.t
            self.t, self.x = self.get_t_and_x()
            delta_t = self.t - previous_t

            # print(f'iteration {iteration} \t t {self.t} \t delta {self.delta()}')

        return last_change and iteration != self.num_iterations

    def solve(self) -> None:
        success = self.inner_solve()
        self.restarts = 0
        while not success and self.restarts < self.max_num_restarts:
            self.restarts += 1
            success = self.inner_solve()
