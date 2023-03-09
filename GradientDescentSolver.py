import numpy as np
import typing as tp
from scipy.stats import special_ortho_group
from utils import solve_primal


class GradientDescentSolver:
    def __init__(self, dimension: int, support_with_argmax_a: tp.Callable, support_with_argmax_b: tp.Callable,
                 max_num_iterations: int = 1000, learning_rate = 0.1, max_finding_distinct_minimums_attempts = 100,
                 tolerance_gd = 1e-10, tolerance_duplicates = 1e-6) -> None:
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
        self.iteration = 0
        self.max_finding_distinct_minimums_attempts = max_finding_distinct_minimums_attempts

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

    def relax_point(self, point: np.ndarray) -> np.ndarray:
        delta_point = np.infty * np.ones_like(point)
        while not np.all(np.abs(delta_point) < self.tolerance_gd):
            support_a_values, support_a_argmaxes = self.support_with_argmax_a(np.reshape(point, (1, self.dimension)))
            support_b_values, support_b_argmaxes = self.support_with_argmax_b(np.reshape(point, (1, self.dimension)))
            point_gradient = self.x + self.t * support_b_argmaxes[0] - support_a_argmaxes[0]
            previous_point = np.copy(point)
            point -= self.learning_rate * point_gradient
            point /= np.linalg.norm(point)
            delta_point = point - previous_point
        return point

    def relax_points(self, points: np.ndarray) -> np.ndarray:
        delta_points = np.infty * np.ones_like(points)
        while not np.all(np.abs(delta_points) < self.tolerance_gd):
            support_a_values, support_a_argmaxes = self.support_with_argmax_a(points)
            support_b_values, support_b_argmaxes = self.support_with_argmax_b(points)
            points_gradient = np.tile(self.x, (len(points), 1)) + self.t * support_b_argmaxes - support_a_argmaxes
            previous_points = np.copy(points)
            points -= self.learning_rate * points_gradient
            points = (points.T / np.linalg.norm(points, axis=1)).T
            delta_points = points - previous_points
        return points

    def omit_duplicates_in_points(self, points: np.ndarray) -> np.ndarray:
        indices_to_delete = []
        for i in range(len(points)):
            for j in range(i):
                if np.all(np.abs(points[i] - points[j]) < self.tolerance_duplicates):
                    indices_to_delete.append(j)
                    #print(f'found duplicates: {points[i]}, {points[j]}')
        return np.delete(points, indices_to_delete, axis=0)

    def find_minimums(self) -> np.ndarray:
        """
        :return: at least dim+1 minimums of (x, p) + t * s(p, B) - s(p, A)
        """
        minimums = self.relax_points(self.grid)
        minimums = self.omit_duplicates_in_points(minimums)
        #print(f'first attempt minimums: \n {minimums}')
        if len(minimums) >= self.dimension + 1:
            return minimums

        minimums = np.vstack((minimums, self.relax_points(- minimums)))
        minimums = self.omit_duplicates_in_points(minimums)
        #print(f'second attempt minimums: \n {minimums}')
        if len(minimums) >= self.dimension + 1:
            return minimums

        for _ in range(self.max_finding_distinct_minimums_attempts):
            minimums = np.vstack((minimums, self.relax_points(self.regular_simplex() @
                                                              special_ortho_group.rvs(dim=self.dimension, size=1))))
            minimums = self.omit_duplicates_in_points(minimums)
            #print(f'{_+2} attempt minimums: \n {minimums}')
            if len(minimums) >= self.dimension + 1:
                return minimums

        return np.vstack((minimums, self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)))

    def update_grid_t_x(self) -> None:
        self.grid = self.find_minimums()
        #print(f'after relaxing the grid is \n {self.grid}')
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)

        if len(self.grid) == self.dimension + 1:
            self.t, self.x = self.get_t_and_x(support_a_values, support_b_values)
        else:
            self.t, self.x = solve_primal(self.grid, support_a_values, support_b_values)
            self.grid = self.grid[self.get_based_gridpoints_indices(support_a_values, support_b_values)]

    def solve(self):
        self.grid = self.regular_simplex() @ special_ortho_group.rvs(dim=self.dimension, size=1)
        support_a_values, support_a_argmaxes = self.support_with_argmax_a(self.grid)
        support_b_values, support_b_argmaxes = self.support_with_argmax_b(self.grid)
        self.t, self.x = self.get_t_and_x(support_a_values, support_b_values)

        delta_t = np.inf
        while self.iteration < self.max_num_iterations and np.abs(delta_t) > self.tolerance_gd:
            self.iteration += 1
            previous_t = self.t
            self.update_grid_t_x()
            delta_t = self.t - previous_t

            #print(f'iteration {self.iteration}\t t {self.t}\t x {self.x}')
