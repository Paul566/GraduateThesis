import numpy as np
import typing as tp
import sys
from scipy.optimize import linprog


def solve_primal(grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray) -> \
        tp.Tuple[tp.Optional[float], tp.Optional[np.ndarray]]:
    q_0 = np.array([0] * len(grid[0]) + [1])
    a_ub = np.hstack((grid, support_b_values.reshape((len(grid), 1))))
    result = linprog(q_0, A_ub=-a_ub, b_ub=-support_a_values, method='highs', bounds=(None, None))
    try:
        t, x = result.x[-1], result.x[:-1]
    except TypeError as e:
        sys.stderr.write(f'linprog status {result.status}')
        raise e

    return t, x


def random_spherical_point(dimension: int):
    """
    :param dimension:
    :return: a uniformly distributed point on a (dimension - 1)-dimensional unit sphere
    """
    x = np.random.normal(size=dimension)
    norm = np.linalg.norm(x)
    if norm == 0.:  # almost never happens
        return np.ones(dimension) / np.sqrt(dimension)
    return x / np.linalg.norm(x)


def random_spherical_grid(dimension: int, grid_size: int) -> np.ndarray:
    """
    :param dimension:
    :param grid_size:
    :return: a grid of random points on a (dimension - 1)-dimensional unit sphere united with +- basis vectors
    """
    return np.vstack((np.identity(dimension),  # this is done to make sure that the convex
                    - np.identity(dimension),    # hull of the gridpoints contains zero
                    np.array([random_spherical_point(dimension) for _ in range(grid_size)])))


def random_ball_point(dimension: int) -> np.ndarray:
    """
    :param dimension:
    :return: a uniformly distributed point on a dimension-dimensional unit ball
    """
    sphere_point = random_spherical_point(dimension)
    r = np.power(np.random.uniform(0, 1), 1. / dimension)
    return r * sphere_point


def random_ball_grid(dimension: int, grid_size: int) -> np.ndarray:
    """
    :param dimension:
    :param grid_size:
    :return: a grid of random points on a dimension-dimensional unit ball
    """
    return np.array([random_ball_point(dimension) for _ in range(grid_size)])


def rotation_to_point(point: np.ndarray) -> np.ndarray:
    """
    :param point: a point on a sphere
    :return: a rotation matrix that maps (1, 0, ..., 0) to the given point
    """
    columns = [point]
    for k in range(len(point) - 1):
        next_col = np.zeros(len(point))
        next_col[k] = - columns[-1][k + 1]
        next_col[k + 1] = columns[-1][k]
        if next_col[k] == 0 and next_col[k + 1] == 0:
            next_col[k] = 1
        columns.append(next_col / np.linalg.norm(next_col))
    return np.array(columns).T


def random_spherical_cap_grid(center: np.ndarray, radius: float, grid_size: int) -> np.ndarray:
    """
    :param center: center of the spherical cap
    :param radius: radius of the spherical cap (in radians), should be small
    :param grid_size: the number of the gridpoints
    :return: an (almost) uniformly sampled points from the spherical cap
    """
    dim = len(center)
    rotation = rotation_to_point(center)
    unrotated_grid = random_ball_grid(dim - 1, grid_size)
    first_column = np.ones(grid_size)
    grid = rotation @ np.vstack((first_column, unrotated_grid.T * radius))
    grid = grid / np.linalg.norm(grid, axis=0)
    return grid.T


class BaselineSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable, grid_size: int) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.grid_size = grid_size
        self.x = None
        self.t = None

    def solve(self) -> None:
        grid = random_spherical_grid(self.dimension, self.grid_size)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)


class IterativeSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable,
                 initial_grid_size: int = 100, initial_cap_radius: float = 1, number_of_gridpoints_in_cap: int = 100,
                 max_iteration: int = 1000, tolerance: float = 1e-9) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.initial_grid_size = initial_grid_size
        self.number_of_gridpoints_in_cap = number_of_gridpoints_in_cap
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
                for p, supp_a, supp_b in zip(grid_in_cap, support_a_values_in_cap, support_b_values_in_cap):
                    delta = p @ self.x + supp_b * self.t - supp_a
                    if - self.tolerance < delta < self.tolerance:
                        ans.append(p)
                        found_based_vector = True
                        break

            if not found_based_vector:
                raise Exception(f'Failed to find a based vector with tolerance {self.tolerance}')

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
