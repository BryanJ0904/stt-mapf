#!/usr/bin/env python3
"""
Simple grid visualizer for STT-CBS / MAPF experiments.

- Reads scenario YAML in the format:
    GRID_SIZE: 1
    ROBOT_RADIUS: 1
    START:
    - !!python/tuple [6, 3]
    ...
    GOAL:
    - !!python/tuple [23, 3]
    ...
    RECT_OBSTACLES:
      0:
      - [0, 0]
      - [100, 100]
      1:
      - [12, 6]
      - [16, 10]

- Displays an n x n grid where n is derived from the maximum coordinate
  appearing in RECT_OBSTACLES, START, and GOAL.
- Animates agent paths where 1 second = 1 timestep.
- Assumes planner output format:
    [
      [[x y], [x y], ...],   # agent 0 path
      [[x y], [x y], ...],   # agent 1 path
      ...
    ]

This file is intentionally independent from the old visualizer.
"""

import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


# ------------------------------------------------------------
# OPTIONAL: uncomment these if you want this file to call planner
# ------------------------------------------------------------
from mapf_stochastic.grid import Grid as MAPF
from stt_cbs_mapf.planner import Planner


class GridAnimationVisualizer:
    def __init__(self, scenario_path: str, paths: List[np.ndarray] = None):
        self.scenario = self.load_scenario(scenario_path)

        self.grid_size = self.scenario["GRID_SIZE"]
        self.robot_radius = self.scenario["ROBOT_RADIUS"]
        self.starts = [tuple(p) for p in self.scenario["START"]]
        self.goals = [tuple(p) for p in self.scenario["GOAL"]]
        self.rect_obstacles = self.scenario["RECT_OBSTACLES"]

        # Build occupancy grid and determine dimension n x n
        self.n = self.compute_grid_dimension()
        self.occupancy = self.build_occupancy_grid()

        # Visual parameters
        self.cell_px = 35
        self.margin = 40
        self.info_bar_h = 60

        self.canvas_w = self.margin * 2 + self.n * self.cell_px
        self.canvas_h = self.margin * 2 + self.n * self.cell_px + self.info_bar_h

        # If no paths provided, use demo fallback from START -> wait
        self.paths = self.normalize_paths(paths if paths is not None else self.make_demo_paths())

        self.agent_colors = self.assign_colors(len(self.paths))
        self.max_steps = max(len(p) for p in self.paths) if self.paths else 0

    # ------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------
    @staticmethod
    def load_scenario(path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def compute_grid_dimension(self) -> int:
        """
        n is based on the maximum coordinate found in:
        - RECT_OBSTACLES corners
        - START
        - GOAL

        If max coordinate is 100, grid becomes 101 x 101 to include cell 100.
        """
        max_coord = 0

        for p in self.starts:
            max_coord = max(max_coord, p[0], p[1])

        for p in self.goals:
            max_coord = max(max_coord, p[0], p[1])

        for _, rect in self.rect_obstacles.items():
            (x0, y0), (x1, y1) = rect
            max_coord = max(max_coord, x0, y0, x1, y1)

        return max_coord + 1

    def build_occupancy_grid(self) -> np.ndarray:
        """
        Create an n x n grid.
        0 = free
        1 = blocked

        Interpretation:
        - RECT_OBSTACLES[0] (or key "0") = boundary rectangle ONLY (block just border cells)
        - all other RECT_OBSTACLES = filled rectangles
        """
        grid = np.zeros((self.n, self.n), dtype=np.uint8)

        for key, rect in self.rect_obstacles.items():
            (x0, y0), (x1, y1) = rect
            xmin, xmax = sorted([int(x0), int(x1)])
            ymin, ymax = sorted([int(y0), int(y1)])

            # Clamp to grid bounds
            xmin = max(0, min(xmin, self.n - 1))
            xmax = max(0, min(xmax, self.n - 1))
            ymin = max(0, min(ymin, self.n - 1))
            ymax = max(0, min(ymax, self.n - 1))

            if str(key) == "0":
                # boundary only
                grid[ymin, xmin:xmax + 1] = 1
                grid[ymax, xmin:xmax + 1] = 1
                grid[ymin:ymax + 1, xmin] = 1
                grid[ymin:ymax + 1, xmax] = 1
            else:
                # filled obstacle
                grid[ymin:ymax + 1, xmin:xmax + 1] = 1
        return grid

    # ------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------
    @staticmethod
    def normalize_paths(paths: List) -> List[np.ndarray]:
        """
        Convert planner output into list of np.ndarray with shape (T, 2).
        Accepts lists or numpy arrays.
        """
        normalized = []
        for p in paths:
            arr = np.array(p, dtype=int)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"Invalid path shape: {arr.shape}. Expected (T, 2).")
            normalized.append(arr)
        return normalized

    def make_demo_paths(self) -> List[np.ndarray]:
        """
        Fallback demo if no planner paths are provided.
        Each agent stays at START.
        """
        demo = []
        for s in self.starts:
            demo.append(np.array([s for _ in range(5)], dtype=int))
        return demo

    # ------------------------------------------------------------
    # Optional planner hook (if you want this file to plan directly)
    # ------------------------------------------------------------
    def plan_with_stt_cbs(self):
        static_obstacles = self.rectangles_to_blocked_points()
        mapf = MAPF(
            grid_size=self.grid_size,
            robot_radius=self.robot_radius,
            static_obstacles=static_obstacles,
            n_v=1.0,
            lambda_v=3.0,
        )
        planner = Planner(grid=mapf)
        raw_paths = planner.plan(self.starts, self.goals, epsilon=0.01, debug=False)
        self.paths = self.normalize_paths(raw_paths)
        self.max_steps = max(len(p) for p in self.paths)
        print('Solution found!')

    def rectangles_to_blocked_points(self):
        """
        Convert RECT_OBSTACLES into blocked grid points.

        Semantics:
        - RECT_OBSTACLES[0] (or key "0") = boundary only
          -> only border cells are blocked
        - all other RECT_OBSTACLES = filled rectangles
          -> all cells inside are blocked
        """
        pts = set()  # use set to avoid duplicates

        for key, rect in self.rect_obstacles.items():
            (x0, y0), (x1, y1) = rect
            xmin, xmax = sorted([int(x0), int(x1)])
            ymin, ymax = sorted([int(y0), int(y1)])

            # Clamp to grid bounds if self.n exists
            xmin = max(0, min(xmin, self.n - 1))
            xmax = max(0, min(xmax, self.n - 1))
            ymin = max(0, min(ymin, self.n - 1))
            ymax = max(0, min(ymax, self.n - 1))

            if str(key) == "0":
                # Boundary only
                pts.add((xmin, ymin))
                pts.add((xmax, ymax))

                # for x in range(xmin, xmax + 1):
                #     pts.add((x, ymin))  # top edge
                #     pts.add((x, ymax))  # bottom edge
                #
                # for y in range(ymin, ymax + 1):
                #     pts.add((xmin, y))  # left edge
                #     pts.add((xmax, y))  # right edge
            else:
                # Filled obstacle: add all interior points
                for y in range(ymin, ymax + 1):
                    for x in range(xmin, xmax + 1):
                        pts.add((x, y))
        return list(pts)

    # ------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------
    @staticmethod
    def assign_colors(num: int) -> Dict[int, Tuple[int, int, int]]:
        def color(i):
            x = hash(str(i + 42))
            return (x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF)

        return {i: color(i) for i in range(num)}

    def cell_top_left(self, x: int, y: int) -> Tuple[int, int]:
        px = self.margin + x * self.cell_px
        py = self.margin + y * self.cell_px
        return px, py

    def cell_center(self, x: int, y: int) -> Tuple[int, int]:
        px, py = self.cell_top_left(x, y)
        return px + self.cell_px // 2, py + self.cell_px // 2

    def draw_grid(self, frame: np.ndarray) -> None:
        # Fill blocked cells first
        for y in range(self.n):
            for x in range(self.n):
                if self.occupancy[y, x] == 1:
                    tl = self.cell_top_left(x, y)
                    br = (tl[0] + self.cell_px, tl[1] + self.cell_px)
                    cv2.rectangle(frame, tl, br, (80, 80, 80), -1)

        # Draw grid lines
        for i in range(self.n + 1):
            # vertical
            x = self.margin + i * self.cell_px
            cv2.line(
                frame,
                (x, self.margin),
                (x, self.margin + self.n * self.cell_px),
                (200, 200, 200),
                1,
            )
            # horizontal
            y = self.margin + i * self.cell_px
            cv2.line(
                frame,
                (self.margin, y),
                (self.margin + self.n * self.cell_px, y),
                (200, 200, 200),
                1,
            )

    def draw_starts_goals(self, frame: np.ndarray) -> None:
        # START = small filled circle
        for i, (x, y) in enumerate(self.starts):
            c = self.cell_center(x, y)
            cv2.circle(frame, c, max(3, self.cell_px // 6), self.agent_colors.get(i, (0, 0, 0)), -1)

        # GOAL = outlined square
        for i, (x, y) in enumerate(self.goals):
            tl = self.cell_top_left(x, y)
            pad = max(3, self.cell_px // 5)
            p0 = (tl[0] + pad, tl[1] + pad)
            p1 = (tl[0] + self.cell_px - pad, tl[1] + self.cell_px - pad)
            cv2.rectangle(frame, p0, p1, self.agent_colors.get(i, (0, 0, 0)), 2)

    def draw_agent_paths(self, frame: np.ndarray, upto_step: int) -> None:
        """
        Draw trail up to current timestep.
        """
        for i, path in enumerate(self.paths):
            c = self.agent_colors[i]
            last = min(upto_step, len(path) - 1)
            for t in range(last + 1):
                x, y = path[t]
                center = self.cell_center(int(x), int(y))
                cv2.circle(frame, center, max(2, self.cell_px // 8), c, -1)

    def draw_agents(self, frame: np.ndarray, step: int) -> None:
        for i, path in enumerate(self.paths):
            idx = min(step, len(path) - 1)
            x, y = path[idx]
            center = self.cell_center(int(x), int(y))
            radius = max(6, self.cell_px // 3)
            cv2.circle(frame, center, radius, self.agent_colors[i], 2)
            cv2.putText(
                frame,
                str(i),
                (center[0] - 8, center[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.agent_colors[i],
                1,
                cv2.LINE_AA,
            )

    def draw_info(self, frame: np.ndarray, step: int) -> None:
        text = f"Step: {step}/{max(0, self.max_steps - 1)}   (1 second = 1 timestep)   Press q to quit"
        cv2.putText(
            frame,
            text,
            (self.margin, self.canvas_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )

    def render_frame(self, step: int) -> np.ndarray:
        frame = np.ones((self.canvas_h, self.canvas_w, 3), dtype=np.uint8) * 255
        self.draw_grid(frame)
        self.draw_starts_goals(frame)
        self.draw_agent_paths(frame, step)
        self.draw_agents(frame, step)
        self.draw_info(frame, step)
        return frame

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
    def animate(self):
        cv2.namedWindow("Grid Animation", cv2.WINDOW_NORMAL)

        # Resize display window if huge
        show_w = min(1400, self.canvas_w)
        show_h = min(900, self.canvas_h)
        cv2.resizeWindow("Grid Animation", show_w, show_h)

        step = 0
        paused_on_first = True

        while True:
            frame = self.render_frame(step)
            cv2.imshow("Grid Animation", frame)

            if paused_on_first:
                # Wait for any key before starting animation
                k = cv2.waitKey(0) & 0xFF
                paused_on_first = False
                if k == ord('q'):
                    break
            else:
                # 1000 ms = 1 second = 1 timestep
                k = cv2.waitKey(1000) & 0xFF
                if k == ord('q'):
                    break
                step += 1
                if step >= self.max_steps:
                    step = self.max_steps - 1

        cv2.destroyAllWindows()


# ------------------------------------------------------------
# Example direct usage with your sample path format
# ------------------------------------------------------------
def example_paths():
    return [
        np.array([
            [5, 3],
            [5, 4],
            [4, 4],
            [4, 5],
            [4, 6],
            [5, 6],
            [5, 6],
        ]),
        np.array([
            [4, 5],
            [5, 5],
            [6, 5],
            [6, 5],
            [6, 5],
            [6, 5],
            [6, 5],
        ]),
        np.array([
            [7, 3],
            [6, 3],
            [6, 2],
            [5, 2],
            [4, 2],
            [3, 2],
            [3, 3],
        ]),
    ]


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python grid_animation_visualizer.py scenario.yaml")
    #     sys.exit(1)
    #
    # scenario_file = sys.argv[1]

    # --------------------------------------------------------
    # OPTION 1: Use example paths (as requested format example)
    # --------------------------------------------------------
    # vis = GridAnimationVisualizer('scenario1.yaml', paths=example_paths())

    # --------------------------------------------------------
    # OPTION 2: If you want to integrate planner directly later:
    vis = GridAnimationVisualizer('scenario1.yaml')
    vis.plan_with_stt_cbs()
    # --------------------------------------------------------

    vis.animate()