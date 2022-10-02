from scipy.optimize import linprog
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime


def support_poly(p, x):
    return np.max(np.dot(x, p))


def solve_primal(grid, support_a_func, support_b_func, method='highs'):
    """
    solves the primal problem
    works poorly, don't use it
    """
    support_a = np.array([support_a_func(p) for p in grid])
    support_b = np.array([support_b_func(p) for p in grid])

    q_0 = np.array([0] * len(grid[0]) + [1])
    a_ub = np.hstack((grid, np.array([[s] for s in support_b])))
    ans = linprog(q_0, A_ub=-a_ub, b_ub=-support_a, method=method).x
    return ans


def solve_dual(grid, support_a_func, support_b_func, method='highs', eps=0.):
    """
    solves the dual problem for the approximated A and B
    """
    support_a = np.array([support_a_func(p) for p in grid])
    support_b = np.array([support_b_func(p) for p in grid])

    q_0 = np.array([0] * len(grid[0]) + [1])
    a_ub = np.hstack((grid, np.array([[s] for s in support_b])))
    lambdas = linprog(-support_a, A_eq=np.transpose(a_ub), b_eq=q_0, method=method).x

    matrix = []
    rhs = []
    for my_lambda, row, s in zip(lambdas, a_ub, support_a):
        if my_lambda > eps:
            matrix.append(row)
            rhs.append(s)

    if len(matrix) != len(matrix[0]):  # sometimes this happens
        print('degeneracy:', len(rhs), 'non-zero lambdas in dimension', len(grid[0]))
        return None

    return np.linalg.solve(matrix, rhs)


def solve_dual_return_lambdas(grid, support_a_func, support_b_func, method='highs'):
    support_a = np.array([support_a_func(p) for p in grid])
    support_b = np.array([support_b_func(p) for p in grid])

    q_0 = np.array([0] * len(grid[0]) + [1])
    a_ub = np.hstack((grid, np.array([[s] for s in support_b])))
    return linprog(-support_a, A_eq=np.transpose(a_ub), b_eq=q_0, method=method).x


'''def iterations_algo_2d(support_a_func, support_b_func, max_iter=100, eps=0.):
    grid = np.transpose((np.cos(np.arange(n) * 2 * np.pi / n), np.sin(np.arange(n) * 2 * np.pi / n)))
    lambdas = solve_dual_return_lambdas(grid, support_a_func, support_b_func)
    non_zero_lambda_indices = []
    matrix = []
    rhs = []
    for my_lambda, row, s, i in zip(lambdas, a_ub, supportA, range(len(lambdas))):
        if my_lambda > eps:
            matrix.append(row)
            rhs.append(s)
            non_zero_lambda_indices.append(i)

    current_solution = np.linalg.solve(matrix, rhs)

    for iteration in range(max_iter):
        grid = list(grid)
        for i in non_zero_lambda_indices:
            grid.append(grid[i] + grid[i - 1])
'''


def example_square(n):
    """
    A = a square in a unit circle rotated by 1 radian
    B = a unit circle
    returns the error of x and t
    """
    grid = np.transpose((np.cos(np.arange(n) * 2 * np.pi / n), np.sin(np.arange(n) * 2 * np.pi / n)))
    vertices_a = np.array([
        [np.cos(1), np.sin(1)],
        [np.cos(1 + np.pi / 2), np.sin(1 + np.pi / 2)],
        [np.cos(1 + np.pi), np.sin(1 + np.pi)],
        [np.cos(1 + np.pi * 3 / 2), np.sin(1 + np.pi * 3 / 2)]
    ])  # A is a rotated square
    ans = solve_dual(grid, lambda x: support_poly(x, vertices_a), lambda x: 1)
    if ans is None:
        return None, None
    return np.sqrt(ans[0] ** 2 + ans[1] ** 2), np.abs(ans[2] - 1)


def example_triangle(n):
    """
    A = a triangle in a unit circle rotated by 1 radian
    B = a unit circle
    returns the error of x and t
    """
    grid = np.transpose((np.cos(np.arange(n) * 2 * np.pi / n), np.sin(np.arange(n) * 2 * np.pi / n)))
    vertices_a = np.array([
        [np.cos(1), np.sin(1)],
        [np.cos(1 + 2 * np.pi / 3), np.sin(1 + 2 * np.pi / 3)],
        [np.cos(1 + 4 * np.pi / 3), np.sin(1 + 4 * np.pi / 3)]
    ])  # A is a rotated square
    ans = solve_primal(grid, lambda x: support_poly(x, vertices_a), lambda x: 1, method='highs')
    if ans is None:
        return None, None
    return np.sqrt(ans[0] ** 2 + ans[1] ** 2), np.abs(ans[2] - 1)


def run_tests(n_start=99, n_end=10000, n_step=100, err_filename='err',
              time_filename='time', example_function=example_square):
    n = range(n_start, n_end, n_step)
    err_x = []
    err_t = []
    times = []

    for _n in n:
        start = time.time()
        _errX, _errT = example_function(_n)
        end = time.time()
        print(_n, '\t', end - start, '\t', _errX, '\t', _errT)
        times.append(end - start)
        err_x.append(_errX)
        err_t.append(_errT)

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('error')
    plt.yscale('log')
    plt.xscale('log')
    ax.scatter(n, err_x, label='error of x')
    ax.scatter(n, err_t, label='error of t')
    ax.plot(n, 1 - np.cos(np.pi / np.array(n)), 'r', label='epsilon')
    plt.legend()
    plt.savefig('figures/' + err_filename + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.png')

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('time, s')
    ax.scatter(n, times)
    plt.savefig('figures/' + time_filename + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.png')


if __name__ == '__main__':
    run_tests(n_step=100, example_function=example_triangle)
