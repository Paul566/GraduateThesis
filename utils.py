import os
import random
import time
import scipy
import numpy as np
import typing as tp
from scipy.optimize import linprog


def solve_primal(grid: np.ndarray, support_a_values: np.ndarray, support_b_values: np.ndarray) -> \
        tp.Tuple[tp.Optional[float], tp.Optional[np.ndarray]]:
    q_0 = np.array([0] * len(grid[0]) + [1])
    a_ub = np.hstack((grid, support_b_values.reshape((len(grid), 1))))
    result = linprog(q_0, A_ub=-a_ub, b_ub=-support_a_values, bounds=(None, None), options={'disp': False})
    return result.x[-1], result.x[:-1]


def sphere_volume(dimension: int):
    """
    :param dimension:
    :return: returns the volume of a (dimension - 1)-dimensional unit sphere (in R^dimension)
    """
    return 2 * np.pi**(dimension / 2) / scipy.special.gamma(dimension / 2)


def ball_volume(dimension: int):
    """
    :param dimension:
    :return: returns the volume of a dimension-dimensional unit ball
    """
    return np.pi**(dimension / 2) / scipy.special.gamma(dimension / 2 + 1)


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


def uniform_spherical_coordinates(dimension: int, num_points_in_pi: int) -> np.ndarray:
    linspaces = [np.linspace(0, np.pi, num_points_in_pi + 1) for _ in range(dimension - 2)]
    linspaces.append(np.linspace(0, 2 * np.pi, 2 * num_points_in_pi + 1))
    coordinate_grids = np.meshgrid(* linspaces)
    return np.vstack([coordinates.flatten() for coordinates in coordinate_grids]).T


def spherical_grid_uniform_coordinates(dimension: int, grid_size: int) -> np.ndarray:
    num_points_in_pi = int((grid_size / 2.) ** (1. / (dimension - 1)))
    phis = uniform_spherical_coordinates(dimension, num_points_in_pi)
    cos_parts = np.hstack((np.cos(phis), np.ones((len(phis), 1))))
    sin_parts = np.hstack((np.ones((len(phis), 1)), np.cumprod(np.sin(phis), axis=1)))
    return sin_parts * cos_parts


def ball_grid_uniform_spherical_coordinates(dimension: int, grid_size: int) -> np.ndarray:
    num_points_in_pi = int((grid_size * np.pi / 2) ** (1. / dimension))
    num_radii = int(num_points_in_pi / np.pi)
    sphere_grids = []   # todo: rewrite avoiding loops
    for r in np.linspace(0, 1, num_radii + 1):
        print(r)
        sphere_grids.append(r * spherical_grid_uniform_coordinates(dimension, int(num_points_in_pi * (r ** (dimension - 1)))))
    return np.vstack(sphere_grids)


def spherical_cap_grid_uniform_coordinates(center: np.ndarray, radius: float, grid_size: int) -> np.ndarray:
    """
    :param center: center of the spherical cap
    :param radius: radius of the spherical cap (in radians), should be small
    :param grid_size: the number of the gridpoints
    :return: a grid on the spherical cap, spherical coordinates on the cap as a dim-1 - dimensional ball are uniform
    """
    dim = len(center)
    rotation = rotation_to_point(center)
    unrotated_grid = ball_grid_uniform_spherical_coordinates(dim - 1, grid_size)
    first_column = np.ones(grid_size)
    grid = rotation @ np.vstack((first_column, unrotated_grid.T * radius))
    grid = grid / np.linalg.norm(grid, axis=0)
    return grid.T


def sphere_grid_from_cube(dimension: int, num_points_in_edge: int) -> np.ndarray:
    """
    :param dimension:
    :param num_points_in_edge:
    :return: a grid with (2 * dimension * num_points_in_edge ** (dimension - 1)) gridpoints on a unit sphere
                obtained via normalization of a grid on a surface of a cube
    """

    grids_of_cube_faces = []
    for i in range(dimension):
        linspaces = [np.linspace(- 1, 1, num_points_in_edge + 1) for _ in range(i)] + \
                    [np.array([1])] + \
                    [np.linspace(- 1, 1, num_points_in_edge + 1) for _ in range(dimension - i - 1)]
        coordinate_grids = np.meshgrid(*linspaces)
        grid_of_cube_face = np.vstack([coordinates.flatten() for coordinates in coordinate_grids]).T
        grids_of_cube_faces.append(grid_of_cube_face)
        grids_of_cube_faces.append(- grid_of_cube_face)

    cube_grid = np.vstack(grids_of_cube_faces)

    return np.unique(cube_grid / np.linalg.norm(cube_grid, axis=1)[:, np.newaxis], axis=0)


def ball_grid_from_cube(dimension: int, num_points_in_edge: int) -> np.ndarray:
    """
        :param dimension:
        :param num_points_in_edge:
        :return: a grid with (num_points_in_edge ** dimension) gridpoints in a unit ball
                    obtained via normalization of a grid in a cube
        """

    linspaces = [np.linspace(- 1, 1, num_points_in_edge) for _ in range(dimension)]
    coordinate_grids = np.meshgrid(*linspaces)
    grid_in_cube = np.vstack([coordinates.flatten() for coordinates in coordinate_grids]).T

    if len(grid_in_cube) % 2 == 1:  # if there is (0, ..., 0) point which is not normalizeable
        tol = 1e-12
        grid_in_cube[len(grid_in_cube) // 2][0] += tol

    return grid_in_cube / np.linalg.norm(grid_in_cube, axis=1)[:, np.newaxis] * \
        np.linalg.norm(grid_in_cube, axis=1, ord=np.inf)[:, np.newaxis]


def spherical_cap_grid_from_cube(center: np.ndarray, radius: float, num_points_in_edge: int) -> np.ndarray:
    """
    :param center: center of the spherical cap
    :param radius: radius of the spherical cap (in radians), should be small
    :param num_points_in_edge: the number of the gridpoints in the diameter of the cap
    :return: a grid on the spherical cap, via creating a grid in a (dimension - 1)-dimensional cube
    """
    dim = len(center)
    rotation = rotation_to_point(center)
    unrotated_grid = ball_grid_from_cube(dim - 1, num_points_in_edge)
    first_column = np.ones(len(unrotated_grid))
    grid = rotation @ np.vstack((first_column, unrotated_grid.T * radius))
    grid = (grid / np.linalg.norm(grid, axis=0)).T
    grid = np.vstack((grid, center)) # add the center of the cap to the grid in case it was a good based point
    return grid


def read_tests_simplex_in_ball(path: str) -> tp.Tuple[tp.Callable, tp.Callable]:
    vertices = np.genfromtxt(path, delimiter=',')
    support_a = lambda grid: np.max(vertices @ grid.T, axis=0)
    support_b = lambda grid: np.ones(len(grid))
    return support_a, support_b


def run_tests(solver: tp.Type, dimension: int, solver_kwargs: tp.Dict[str, tp.Any], test_reader_function: tp.Callable,
              path_to_tests: str, silent: bool=False) -> tp.Tuple[np.ndarray, np.ndarray]:
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
        if not silent:
            print(f'test \t{file}\t time \t{end_time - start_time}\t t_error \t{abs(1. - solver_instance.t)}')

    return np.array(times), np.array(t_accuracies)


def run_random_test(solver: tp.Type, dimension: int, solver_kwargs: tp.Dict[str, tp.Any], test_reader_function: tp.Callable,
              path_to_tests: str, silent: bool=False) -> tp.Tuple[float, float]:
    file = random.choice(os.listdir(path_to_tests))
    support_a, support_b = test_reader_function(f'{path_to_tests}/{file}')
    solver_instance = solver(dimension, support_a, support_b, **solver_kwargs)

    start_time = time.time()
    solver_instance.solve()
    end_time = time.time()

    if not silent:
        print(f'test \t{file}\t time \t{end_time - start_time}\t t_error \t{abs(1. - solver_instance.t)}')
    return end_time - start_time, abs(1. - solver_instance.t)






