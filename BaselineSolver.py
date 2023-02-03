import typing as tp
from utils import random_spherical_grid, solve_primal, sphere_volume


class BaselineSolver:
    def __init__(self, dimension: int, support_a: tp.Callable, support_b: tp.Callable, grid_density: float = 100) -> None:
        self.support_a = support_a
        self.support_b = support_b
        self.dimension = dimension
        self.grid_size = int(sphere_volume(dimension) * grid_density)
        self.x = None
        self.t = None

    def solve(self) -> None:
        grid = random_spherical_grid(self.dimension, self.grid_size)
        support_a_values = self.support_a(grid)
        support_b_values = self.support_b(grid)
        self.t, self.x = solve_primal(grid, support_a_values, support_b_values)
