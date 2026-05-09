#!/usr/bin/env python3
"""
[REFactored for Stochastic MAPF]
Author: Haoran Peng
Email: gavinsweden@gmail.com

Grid environment for MAPF with stochastic delay modeling using Gamma distribution.

Gamma parameterization used:
    shape = n_v
    rate  = lambda_v

Equivalent NumPy sampling uses:
    scale = 1 / lambda_v

Thus:
    delay ~ Gamma(shape=n_v, scale=1/lambda_v)

Expected delay:
    E[delay] = n_v / lambda_v

Variance:
    Var[delay] = n_v / (lambda_v^2)
"""

from typing import Tuple, List, Dict, Set, Optional, Union
from visualization.interface import Interface
import numpy as np

class Grid:
    """
    Grid environment for MAPF with optional stochastic delay per vertex.

    Parameters
    ----------
    grid_size : int
        Size of each grid cell.
    static_obstacles : np.ndarray
        Array of obstacle coordinates used to infer environment boundaries.
        Expected shape: (N, 2)
    n_v : float or np.ndarray, optional
        Gamma shape parameter for stochastic delay.
        - If scalar: same parameter for all vertices.
        - If ndarray: shape must match grid dimensions (rows, cols).
    lambda_v : float or np.ndarray, optional
        Gamma rate parameter for stochastic delay.
        - If scalar: same parameter for all vertices.
        - If ndarray: shape must match grid dimensions (rows, cols).
    """

    def __init__(
        self,
        scenario: str
    ):

        self.interface = Interface(scenario)

        self.grid_size = self.interface.grid_size
        self.robot_radius = self.interface.robot_radius
        self.static_obstacles = np.array(self.interface.static_obstacles)

        self.minx, self.maxx, self.miny, self.maxy = self.calculate_boundaries(self.static_obstacles)
        self.grid = self.make_grid(self.interface.grid_size, self.minx, self.maxx, self.miny, self.maxy)

        # Grid dimensions
        self.rows = self.grid.shape[0]
        self.cols = self.grid.shape[1]

        # Gamma parameters per vertex
        self.n_v = self._initialize_param_map(self.interface.n_v, "n_v")
        self.lambda_v = self._initialize_param_map(self.interface.lambda_v, "lambda_v")

        self._validate_gamma_params()

    @staticmethod
    def calculate_boundaries(static_obstacles: np.ndarray) -> Tuple[int, int, int, int]:
        min_ = np.min(static_obstacles, axis=0)
        max_ = np.max(static_obstacles, axis=0)
        return min_[0], max_[0], min_[1], max_[1]

    @staticmethod
    def make_grid(grid_size: int, minx: int, maxx: int, miny: int, maxy: int) -> np.ndarray:
        """
        Create grid cell centers.

        Returns
        -------
        np.ndarray
            Shape: (rows, cols, 2)
            Each entry contains [x_center, y_center]
        """
        # +1 ensures the boundary cells are included
        x_size = max(1, ((maxx - minx) // grid_size) + 1)
        y_size = max(1, ((maxy - miny) // grid_size) + 1)

        grid = np.zeros((y_size, x_size, 2), dtype=np.int32)

        y = miny - grid_size / 2
        for i in range(y_size):
            y += grid_size
            x = minx - grid_size / 2
            for j in range(x_size):
                x += grid_size
                grid[i, j] = np.array([x, y], dtype=np.int32)

        return grid

    def _initialize_param_map(
        self,
        param: Optional[Union[float, np.ndarray]],
        name: str
    ) -> np.ndarray:
        """
        Convert scalar or ndarray parameter into per-vertex map of shape (rows, cols).
        """
        if param is None:
            param = 1.0

        if np.isscalar(param):
            return np.full((self.rows, self.cols), float(param), dtype=np.float64)

        param = np.asarray(param, dtype=np.float64)
        if param.shape != (self.rows, self.cols):
            raise ValueError(
                f"{name} must be scalar or have shape {(self.rows, self.cols)}, "
                f"but got {param.shape}"
            )
        return param

    def _validate_gamma_params(self) -> None:
        """
        Validate Gamma(shape=n_v, rate=lambda_v) parameters.
        """
        if np.any(self.n_v <= 0):
            raise ValueError("All n_v values must be > 0 for Gamma distribution.")
        if np.any(self.lambda_v <= 0):
            raise ValueError("All lambda_v values must be > 0 for Gamma distribution.")

    def position_to_index(self, position: np.ndarray) -> Tuple[int, int]:
        """
        Convert continuous position to grid index (row, col).
        """
        i = int((position[1] - self.miny) // self.grid_size)
        j = int((position[0] - self.minx) // self.grid_size)

        i = max(0, min(i, self.rows - 1))
        j = max(0, min(j, self.cols - 1))

        return i, j

    def index_to_position(self, i: int, j: int) -> np.ndarray:
        """
        Convert grid index (row, col) to cell center position.
        """
        return self.grid[i, j]

    def snap_to_grid(self, position: np.ndarray) -> np.ndarray:
        i = (position[1] - self.miny) // self.grid_size
        j = (position[0] - self.minx) // self.grid_size
        if i >= len(self.grid):
            i -= 1
        if j >= len(self.grid[0]):
            j -= 1
        return self.grid[i][j]

    def get_vertex_delay_params(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Get Gamma parameters (n_v, lambda_v) for a given vertex position.
        """
        i, j = self.position_to_index(position)
        return self.n_v[i, j], self.lambda_v[i, j]

    def sample_vertex_delay(self, position: np.ndarray) -> float:
        """
        Sample stochastic delay at a given vertex using Gamma distribution.

        Gamma parameterization:
            shape = n_v
            rate  = lambda_v

        NumPy uses:
            gamma(shape, scale)
        where:
            scale = 1 / lambda_v
        """
        i, j = self.position_to_index(position)
        shape = self.n_v[i, j]
        rate = self.lambda_v[i, j]
        scale = 1.0 / rate

        return float(np.random.gamma(shape=shape, scale=scale))

    def vertex_delay_by_index(self, i: int, j: int) -> float:
        """
        Sample stochastic delay directly using grid index.
        """
        shape = self.n_v[i, j]
        rate = self.lambda_v[i, j]
        scale = 1.0 / rate

        return float(np.random.gamma(shape=shape, scale=scale))

    def sample_vertex_delays(self, shape: float, rate: float, samples: int = 5000) -> float:
        """
        Sample stochastic delay directly using grid index.
        """
        scale = 1.0 / rate

        return np.random.gamma(shape=shape, scale=scale, size=samples)

    def expected_vertex_delay(self, position: np.ndarray) -> float:
        """
        Expected delay E[X] = n_v / lambda_v for a given vertex.
        """
        i, j = self.position_to_index(position)
        return float(self.n_v[i, j] / self.lambda_v[i, j])

    def variance_vertex_delay(self, position: np.ndarray) -> float:
        """
        Variance Var[X] = n_v / lambda_v^2 for a given vertex.
        """
        i, j = self.position_to_index(position)
        return float(self.n_v[i, j] / (self.lambda_v[i, j] ** 2))

    def sample_delay_map(self) -> np.ndarray:
        """
        Sample one stochastic delay realization for every vertex in the grid.

        Returns
        -------
        np.ndarray
            Delay map of shape (rows, cols)
        """
        scale_map = 1.0 / self.lambda_v
        return np.random.gamma(shape=self.n_v, scale=scale_map)

    def get_expected_delay_map(self) -> np.ndarray:
        """
        Return expected delay map for all vertices.
        """
        return self.n_v / self.lambda_v

    def get_variance_delay_map(self) -> np.ndarray:
        """
        Return variance delay map for all vertices.
        """
        return self.n_v / (self.lambda_v ** 2)

        # ------------------------------------------------------------
        # Planner
        # ------------------------------------------------------------

if __name__ == '__main__':
    # ==========================================================
    # TEST: CHECK GRID SHAPE AND CONTENT
    # ==========================================================
    # Boundary inferred from these points:
    # min = (0, 0), max = (10, 10)
    grid = Grid("original.yaml")

    print("=" * 60)
    print("GRID INFORMATION")
    print("=" * 60)
    print(f"Grid size       : {grid.grid_size}")
    print(f"Boundary X      : [{grid.minx}, {grid.maxx}]")
    print(f"Boundary Y      : [{grid.miny}, {grid.maxy}]")
    print(f"Grid shape      : {grid.grid.shape}")
    print(f"Rows x Cols     : {grid.rows} x {grid.cols}")
    print(f"Static Obstacles: {grid.static_obstacles}")

    print("\n" + "=" * 60)
    print("GRID CELL CENTERS")
    print("=" * 60)
    print(grid.grid)

    print("\n" + "=" * 60)
    print("EXPECTED DELAY MAP")
    print("=" * 60)
    print(grid.get_expected_delay_map())

    print("\n" + "=" * 60)
    print("SAMPLED DELAY MAP (Gamma)")
    print("=" * 60)
    print(grid.sample_delay_map())

    print("\n" + "=" * 60)
    print("SINGLE VERTEX DELAY SAMPLE")
    print("=" * 60)
    print(f"Delay at {np.array([0, 0])}: {grid.sample_vertex_delay(np.array([0, 0])):.4f}")
    print("Sample vertex delays: ", grid.sample_vertex_delays(0.1, 1.0))

    vis = grid.interface
    # vis.plan_with_stt_cbs(grid)
    vis.plan_with_mappo_model()
    vis.animate()