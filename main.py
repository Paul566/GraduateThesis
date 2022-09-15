from scipy.optimize import linprog
import numpy as np
import time
import matplotlib.pyplot as plt


def supportPoly(p, x):
    return np.max(np.dot(x, p))

def testPrimal(n, method='highs'):
    '''
    solves the primal problem
    works poorly, don't use it
    '''
    grid = np.transpose((np.cos(np.arange(n) * 2 * np.pi / n), np.sin(np.arange(n) * 2 * np.pi / n)))
    verticesA = np.array([[-0.5, 0], [0.5, 0], [0, np.sqrt(3.) / 2]])  # A is a triangle
    supportA = np.array([supportPoly(p, verticesA) for p in grid])
    supportB = np.ones(n)  # B is a unit ball
    q_0 = np.array([0, 0, 1])
    a_ub = np.hstack((grid, np.array([[s] for s in supportB])))
    ans = linprog(q_0, A_ub=-a_ub, b_ub=-supportA, method=method)
    if ans.x is None:
        print('answer is none', n, method)
        return None, None
    return np.sqrt(ans.x[0]**2 + (ans.x[1] - np.sqrt(3) / 6)**2), np.abs(ans.x[2] - np.sqrt(3) / 3)

def solveDual(grid, supportA, supportB, method='highs', eps=0.):
    '''
    solves the dual problem for the approximated A and B
    '''
    q_0 = np.array([0, 0, 1])
    a_ub = np.hstack((grid, np.array([[s] for s in supportB])))
    lambdas = linprog(-supportA, A_eq=np.transpose(a_ub), b_eq=q_0, method=method).x

    matrix = []
    rhs = []
    for l, row, s in zip(lambdas, a_ub, supportA):
        if l > eps:
            matrix.append(row)
            rhs.append(s)

    if len(matrix) == len(matrix[0]) - 1:   #sometimes this happens, have to make matrix square
        matrix.append(np.zeros(len(matrix[0])))
        rhs.append(0)
        #for l, row, s in zip(lambdas, a_ub, supportA):
        #    if l <= eps:
        #        matrix[-1] += row
        #        rhs += s

    return np.linalg.solve(matrix, rhs)

def exampleSquare(n):
    '''
    A = a square in a unit circle rotated by 1 radian
    B = a unit circle
    returns the error of x and t
    '''
    grid = np.transpose((np.cos(np.arange(n) * 2 * np.pi / n), np.sin(np.arange(n) * 2 * np.pi / n)))
    verticesA = np.array([
                            [np.cos(1), np.sin(1)], 
                            [np.cos(1 + np.pi / 2), np.sin(1 + np.pi / 2)], 
                            [np.cos(1 + np.pi), np.sin(1 + np.pi)], 
                            [np.cos(1 + np.pi * 3 / 2), np.sin(1 + np.pi * 3 / 2)]
                        ])  # A is a rotated square
    supportA = np.array([supportPoly(p, verticesA) for p in grid])
    supportB = np.ones(n)  # B is a unit ball
    ans = solveDual(grid, supportA, supportB)
    return np.sqrt(ans[0]**2 + ans[1]**2), np.abs(ans[2] - 1)


if __name__ == '__main__':
    n = range(99, 10000, 100)
    errX = []
    errT = []
    times = []

    for _n in n:
        start = time.time()
        _errX, _errT = exampleSquare(_n)
        end = time.time()
        print(_n, end - start)
        times.append(end - start)
        errX.append(_errX)
        errT.append(_errT)

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('error')
    plt.yscale('log')
    plt.xscale('log')
    ax.scatter(n, errX, label='error of x')
    ax.scatter(n, errT, label='error of t')
    ax.plot(n,  1 - np.cos(np.pi / np.array(n)), 'r', label='epsilon')
    plt.legend()
    plt.savefig('err.png')

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('time, s')
    ax.scatter(n, times)
    plt.savefig('time.png')


