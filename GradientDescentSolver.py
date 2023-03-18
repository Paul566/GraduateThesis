import numpy as np
import typing as tp
from scipy.stats import special_ortho_group
from utils import random_spherical_point


class GradientDescentSolver:
    def __init__(self, dimension: int, support_with_argmax_a: tp.Callable, support_with_argmax_b: tp.Callable,
                 max_num_iterations: int = 20, learning_rate = 1,
                 max_finding_distinct_minimums_attempts_per_dimension = 10,
                 max_num_gd_iterations: int = 1000,
                 tolerance_gd = 1e-10, tolerance_duplicates = 1e-6, max_num_restarts = 5) -> None:
        """
        :param dimension:
        :param support_with_argmax_a: should return a tuple of arrays: (support values and argmaxes)
        :param support_with_argmax_b:
        :param max_num_iterations:
        :param learning_rate:
        """
        self.support_with_argmax_a = support_with_argmax_a
        self.support_with_argmax_b = support_with_argmax_b
        self.dimension = dimension
        self.t: float | None = None
        self.x: np.ndarray | None = None
        self.grid: np.ndarray | None = None
        self.max_num_iterations = max_num_iterations
        self.learning_rate = learning_rate
        self.tolerance_gd = tolerance_gd
        self.tolerance_duplicates = tolerance_duplicates
        self.max_finding_distinct_minimums_attempts = max_finding_distinct_minimums_attempts_per_dimension * dimension
        self.max_num_gd_iterations = max_num_gd_iterations
        self.max_num_restarts = max_num_restarts
        self.restart_counter = 0
        self.final_iteration: int | None = None

    def zero_inside_convex_hull(self, simplex):
        new_basis = simplex[1:] - np.tile(simplex[0, :], (self.dimension, 1))
        try:
            zero_in_new_basis = - simplex[0, :] @ np.linalg.inv(new_basis)
        except np.linalg.LinAlgError:  # degenerate case
            return False
        return np.all(zero_in_new_basis > 0) and np.sum(zero_in_new_basis) < 1

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

    def relax_points(self, points: np.ndarray) -> np.ndarray:
        for _ in range(self.max_num_gd_iterations):
            support_a_values, support_a_argmaxes = self.support_with_argmax_a(points)
            support_b_values, support_b_argmaxes = self.support_with_argmax_b(points)
            points_gradient = np.tile(self.x, (len(points), 1)) + self.t * support_b_argmaxes - support_a_argmaxes
            previous_points = np.copy(points)
            points -= self.learning_rate * points_gradient
            points = (points.T / np.linalg.norm(points, axis=1)).T
            delta_points = points - previous_points
            if np.all(np.abs(delta_points) < self.tolerance_gd):
                break
        return points

    def omit_duplicates_in_points(self, points: np.ndarray) -> np.ndarray:
        # TODO: rewrite this without the inner loop
        indices_to_delete = []
        for i in range(len(points)):
            for j in range(i):
                if np.all(np.abs(points[i] - points[j]) < self.tolerance_duplicates):
                    indices_to_delete.append(j)
        return np.delete(points, indices_to_delete, axis=0)

    def find_minimums(self) -> np.ndarray:
        """
        :return: dim+1 minimums of (x, p) + t * s(p, B) - s(p, A), if wasn't able to find dim+1 distinct minimums,
        then complete the found minimums with some points to get a grid such that the size of the grid is n+1,
        and zero is in the convex hull of the grid
        """
        minimums = self.relax_points(self.grid)
        minimums = self.omit_duplicates_in_points(minimums)
        if len(minimums) == self.dimension + 1:
            return minimums

        for _ in range(self.max_finding_distinct_minimums_attempts):
            new_point = random_spherical_point(self.dimension)
            new_point = self.relax_points(np.array([new_point]))[0]

            # if it's a new minimum, then add it to the existing minimums
            if np.all(np.max(np.abs(minimums - np.tile(new_point, (len(minimums), 1))), axis=1) >
                      self.tolerance_duplicates):
                minimums = np.vstack((minimums, new_point))

            if len(minimums) == self.dimension + 1:
                return minimums

        # if we still didn't get the n+1 minimums, add some points to the grid such that
        # the size of the grid is n+1, and zero is in the convex hull of the grid
        if len(minimums) < self.dimension:
            random_points_to_add = np.random.normal(size=(self.dimension - len(minimums), self.dimension))
            random_points_to_add = (random_points_to_add.T / np.linalg.norm(random_points_to_add, axis=1)).T
            minimums = np.vstack((minimums, random_points_to_add))
        final_point = np.sum(minimums, axis=0)
        final_point = - final_point / np.linalg.norm(final_point)
        return np.vstack((minimums, final_point))

    def update_grid_t_x(self) -> None:
        self.grid = self.find_minimums()
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
        self.t, self.x = self.get_t_and_x(support_a_values, support_b_values)

    def solve(self):
        self.grid = self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
        self.t, self.x = self.get_t_and_x(support_a_values, support_b_values)

        for i in range(self.max_num_iterations):
            #print(f'iteration {i}\t t {self.t}\t |x| {np.linalg.norm(self.x)}\t zero in co(grid): {self.zero_inside_convex_hull(self.grid)}')

            previous_t = self.t
            self.update_grid_t_x()

            if np.abs(self.t - previous_t) < self.tolerance_gd:
                if self.zero_inside_convex_hull(self.grid):
                    self.final_iteration = i
                    return
                else:
                    break

        if self.restart_counter < self.max_num_restarts:
            self.restart_counter += 1
            self.solve()