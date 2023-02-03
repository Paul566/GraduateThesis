import logging
import os
import time

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


def read_tests_simplex_in_ball(path: str) -> tp.Tuple[tp.Callable, tp.Callable]:
    vertices = np.genfromtxt(path, delimiter=',')
    support_a = lambda grid: np.max(vertices @ grid.T, axis=0)
    support_b = lambda grid: np.ones(len(grid))
    return support_a, support_b


def run_tests(solver: tp.Type, dimension: int, solver_kwargs: tp.Dict[str, tp.Any],
              test_reader_function: tp.Callable, path_to_tests: str) -> tp.Tuple[np.ndarray, np.ndarray]:
    times = []
    t_accuracies = []
    for file in os.listdir(path_to_tests):
        support_a, support_b = test_reader_function(f'{path_to_tests}/{file}')
        solver_instance = solver(dimension, support_a, support_b, **solver_kwargs)
        start_time = time.time()
        solver_instance.solve()
        end_time = time.time()
        times.append(end_time - start_time)
        t_accuracies.append(abs(1. - solver_instance.t))
        print(f'test \t{file}\t time \t{end_time - start_time}\t t_error \t{abs(1. - solver_instance.t)}')

    return np.array(times), np.array(t_accuracies)






