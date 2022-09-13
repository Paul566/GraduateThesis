from scipy.optimize import linprog
import numpy as np
import time
import matplotlib.pyplot as plt


def supportPoly(p, x):
    return np.max(np.dot(x, p))

def test(n, method='highs'):
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

if __name__ == '__main__':
    n = range(5, 2000, 102)
    errXSimplex = []
    errTSimplex = []
    timesSimplex = []
    errXHighs = []
    errTHighs = []
    timesHighs = []

    for _n in n:
        print(_n)
        start = time.time()
        _errX, _errT = test(_n, method='Highs')
        end = time.time()
        timesHighs.append(end - start)
        errXHighs.append(_errX)
        errTHighs.append(_errT)

        start = time.time()
        _errX, _errT = test(_n, method='Simplex')
        end = time.time()
        timesSimplex.append(end - start)
        errXSimplex.append(_errX)
        errTSimplex.append(_errT)

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('error of t')
    plt.yscale('log')
    ax.scatter(n, errTSimplex, label='method=simplex', marker='+')
    ax.scatter(n, errTHighs, label='method=highs', marker='x')
    plt.legend()
    plt.savefig('errT.png')

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('error of x')
    plt.yscale('log')
    ax.scatter(n, errXSimplex, label='method=simplex', marker='+')
    ax.scatter(n, errXHighs, label='method=highs', marker='x')
    plt.legend()
    plt.savefig('errX.png')

    fig, ax = plt.subplots(1)
    plt.xlabel('n')
    plt.ylabel('time, s')
    plt.yscale('log')
    ax.scatter(n, timesSimplex, label='method=simplex', marker='+')
    ax.scatter(n, timesHighs, label='method=highs', marker='x')
    plt.legend()
    plt.savefig('time.png')


