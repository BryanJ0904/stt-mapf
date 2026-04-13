#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com

An implementation of multi-agent path finding using conflict-based search
[Sharon et al., 2015]
'''
from typing import Dict, Callable, Set, Optional
import multiprocessing as mp
from heapq import heappush, heappop
from itertools import combinations
from copy import deepcopy

# The low level planner for CBS is the Space-Time A* planner
# https://github.com/GavinPHR/Space-Time-AStar
from stastar_stochastic.planner import Planner as STPlanner
from mapf_stochastic.grid import Grid

from .constraint_tree import CTNode
from .constraints import Constraints
from .assigner import *
class Planner:

    def __init__(self, grid: Grid):
        self.robot_radius = grid.robot_radius
        self.st_planner = STPlanner(grid)

    '''
    You can use your own assignment function, the default algorithm greedily assigns
    the closest goal to each start.
    '''
    def plan(self, starts: List[Tuple[int, int]],
                   goals: List[Tuple[int, int]],
                   assign:Callable = by_index,
                   epsilon:int = 0.1,
                   max_iter:int = 200,
                   low_level_max_iter:int = 100,
                   max_process:int = 10,
                   debug:bool = False) -> np.ndarray:

        self.low_level_max_iter = low_level_max_iter
        self.debug = debug

        # Do goal assignment
        self.agents = assign(starts, goals)
        self.p_conflict_threshold = epsilon

        constraints = Constraints()

        # Compute path for each agent using low level planner
        solution = dict((agent, self.calculate_path(agent, constraints, None)) for agent in self.agents)

        open = []
        if all(len(path) != 0 for path in solution.values()):
            # Make root node
            node = CTNode(constraints, solution)
            # Min heap for quick extraction
            open.append(node)

        manager = mp.Manager()
        iter_ = 0
        while open and iter_ < max_iter:
            iter_ += 1

            results = manager.list([])

            processes = []

            # Default to 10 processes maximum
            for _ in range(max_process if len(open) > max_process else len(open)):
                p = mp.Process(target=self.search_node, args=[heappop(open), results])
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for result in results:
                if len(result) == 1:
                    if debug:
                        print('CBS_MAPF: Paths found after about {0} iterations'.format(4 * iter_))
                    return result[0]
                if result[0]:
                    heappush(open, result[0])
                if result[1]:
                    heappush(open, result[1])

        if debug:
            print('CBS-MAPF: Open set is empty, no paths found.')
        return np.array([])

    '''
    Abstracted away the cbs search for multiprocessing.
    The parameters open and results MUST BE of type ListProxy to ensure synchronization.
    '''
    def search_node(self, best: CTNode, results):
        agent_i, agent_j, time_of_conflict_i, time_of_conflict_j = self.validate_paths(self.agents, best)

        # If there is not conflict, validate_paths returns (None, None, -1)
        if agent_i is None:
            results.append((self.reformat(self.agents, best.solution),))
            return
        # Calculate new constraints
        agent_i_constraint = self.calculate_constraints(best, agent_i, agent_j, time_of_conflict_i, time_of_conflict_j)
        agent_j_constraint = self.calculate_constraints(best, agent_j, agent_i, time_of_conflict_j, time_of_conflict_i)

        if self.debug:
            print("Agent i constraint: ", agent_i_constraint)
            print("Agent j constraint: ", agent_j_constraint)
        # Calculate new paths
        agent_i_path = self.calculate_path(agent_i,
                                           agent_i_constraint,
                                           self.calculate_goal_times(best, agent_i, self.agents))
        agent_j_path = self.calculate_path(agent_j,
                                           agent_j_constraint,
                                           self.calculate_goal_times(best, agent_j, self.agents))

        # Replace old paths with new ones in solution
        solution_i = best.solution
        solution_j = deepcopy(best.solution)
        solution_i[agent_i] = agent_i_path
        solution_j[agent_j] = agent_j_path

        node_i = None
        if all(len(path) != 0 for path in solution_i.values()):
            node_i = CTNode(agent_i_constraint, solution_i)

        node_j = None
        if all(len(path) != 0 for path in solution_j.values()):
            node_j = CTNode(agent_j_constraint, solution_j)

        results.append((node_i, node_j))


    '''
    Pair of agent, point of conflict
    '''

    def validate_paths(self, agents, node: CTNode):
        """
        Pairwise validation of all agent paths using probabilistic conflict checking.

        Returns:
            (agent_i, agent_j, conflict_info)
            where conflict_info is a dict containing:
                {
                    "type": "vertex" or "edge",
                    "time": timestep index,
                    "prob": conflict probability
                }

            If no conflict:
                (None, None, -1)
        """
        for agent_i, agent_j in combinations(agents, 2):
            time_of_conflict_i, time_of_conflict_j = self.safe_distance(node.solution, agent_i, agent_j)
            # time_of_conflict=1 if there is not conflict
            if time_of_conflict_i == -1 and time_of_conflict_j == -1:
                continue
            return agent_i, agent_j, time_of_conflict_i, time_of_conflict_j
        return None, None, -1, -1

    # def safe_distance(
    #         self,
    #         solution: Dict[Agent, np.ndarray],
    #         agent_i: Agent,
    #         agent_j: Agent
    # ) -> int:
    #     """
    #     Check probabilistic vertex/edge conflicts between two agents.
    #
    #     A conflict is returned only if:
    #         Pcv >= self.p_conflict_threshold
    #         or
    #         Pce >= self.p_conflict_threshold
    #
    #     Returns:
    #         dict:
    #             {
    #                 "type": "vertex" or "edge",
    #                 "time": timestep index,
    #                 "prob": probability
    #             }
    #         or None if no conflict.
    #     """
    #
    #     path_i = solution[agent_i]
    #     path_j = solution[agent_j]
    #
    #     min_len = min(len(path_i), len(path_j))
    #
    #     # Need at least 1 timestep for vertex conflict
    #     for idx in range(min_len):
    #         point_i = path_i[idx]
    #         point_j = path_j[idx]
    #
    #         # -----------------------------
    #         # 1. Vertex conflict candidate
    #         # -----------------------------
    #         if np.array_equal(path_i[idx], path_j[idx]):
    #             # Planned same timestep => same nominal arrival time
    #             t_diff = 0.0
    #
    #             # Default approximation:
    #             # after idx moves, accumulated delay ~ Gamma(shape = n_i , rate = lambda)
    #             n1 = sum(self.st_planner.grid.n_v.get(tuple(path_i[j]), 1.0) for j in range(idx + 1))
    #             n2 = sum(self.st_planner.grid.n_v.get(tuple(path_j[j]), 1.0) for j in range(idx + 1))
    #
    #             lam = self.st_planner.grid.lambda_v if hasattr(self.st_planner.grid, "lambda_v") else 1.0
    #
    #             p_cv = self.compute_p_conflict_cv(
    #                 t_diff=t_diff,
    #                 n1=n1,
    #                 n2=n2,
    #                 lam=lam
    #             )
    #
    #             if p_cv >= self.p_conflict_threshold:
    #                 # return {
    #                 #     "type": "vertex",
    #                 #     "time": idx,
    #                 #     "prob": p_cv
    #                 # }
    #                 return idx
    #
    #         # -----------------------------
    #         # 2. Edge conflict candidate
    #         # -----------------------------
    #         # Need next timestep for edge traversal
    #         if idx < min_len - 1:
    #             next_i = path_i[idx + 1]
    #             next_j = path_j[idx + 1]
    #
    #             # Check if edges are potentially conflicting:
    #             # classic swap conflict: i: a->b and j: b->a
    #             # You can extend this later for crossing edges in continuous space.
    #             is_swap = np.array_equal(point_i, next_j) and np.array_equal(point_j, next_i)
    #
    #             if is_swap:
    #                 # Same nominal departure timestep
    #                 t_diff = 0.0
    #
    #                 # Approximation: departure from edge at step idx
    #                 n1 = idx + 1
    #                 n2 = idx + 1
    #
    #                 lam = self.lambda_e if hasattr(self, "lambda_e") else (
    #                     self.lambda_v if hasattr(self, "lambda_v") else 1.0
    #                 )
    #                 te = self.edge_time if hasattr(self, "edge_time") else 1.0
    #
    #                 p_ce = self.compute_p_conflict_ce(
    #                     t_diff=t_diff,
    #                     n1=n1,
    #                     n2=n2,
    #                     lam=lam,
    #                     te=te
    #                 )
    #
    #                 if p_ce >= self.p_conflict_threshold:
    #                     # return {
    #                     #     "type": "edge",
    #                     #     "time": idx,
    #                     #     "prob": p_ce
    #                     # }
    #                     return idx
    #
    #     return None

    def safe_distance(
            self,
            solution: Dict[Agent, np.ndarray],
            agent_i: Agent,
            agent_j: Agent
    ) -> int:
        """
        Check probabilistic vertex conflicts between two agents.

        A vertex conflict is checked if both agents ever visit the same vertex,
        even if at different timesteps.

        Returns:
            int: conflict reference timestep (using agent_i timestep) if conflict found
            None : if no conflict
        """

        path_i = solution[agent_i]
        path_j = solution[agent_j]

        # -------------------------------------------------
        # Build vertex -> list of visited timestep indices
        # -------------------------------------------------
        visits_i = {}
        for idx_i, point_i in enumerate(path_i):
            v = tuple(point_i)
            visits_i.setdefault(v, []).append(idx_i)

        visits_j = {}
        for idx_j, point_j in enumerate(path_j):
            v = tuple(point_j)
            visits_j.setdefault(v, []).append(idx_j)

        # Shared vertices (same vertex, different timestep allowed)
        shared_vertices = set(visits_i.keys()) & set(visits_j.keys())

        for v in shared_vertices:
            best_pair = None
            best_tdiff = float("inf")

            # -------------------------------------------------
            # Find the pair of visits with smallest nominal time difference
            # -------------------------------------------------
            for idx_i in visits_i[v]:
                for idx_j in visits_j[v]:
                    # nominal arrival time = timestep index (assuming dt = 1)
                    t_i = idx_i
                    t_j = idx_j

                    tdiff = abs(t_i - t_j)

                    if tdiff < best_tdiff:
                        best_tdiff = tdiff
                        best_pair = (idx_i, idx_j, t_i, t_j)

            if best_pair is None:
                continue

            idx_i, idx_j, t_i, t_j = best_pair

            # -------------------------------------------------
            # Accumulated delay shape up to the shared vertex
            # n1 = sum of n_v along path_i until idx_i
            # n2 = sum of n_v along path_j until idx_j
            # -------------------------------------------------
            n1 = sum(
                self.st_planner.grid.n_v[tuple(path_i[k])]
                for k in range(idx_i + 1)
            )

            n2 = sum(
                self.st_planner.grid.n_v[tuple(path_j[k])]
                for k in range(idx_j + 1)
            )

            # lambda for vertex delay
            lam = self.st_planner.grid.lambda_v[tuple(v)] if hasattr(self.st_planner.grid, "lambda_v") else 1.0

            # local vertex parameter (shared vertex v)
            nv = self.st_planner.grid.n_v[tuple(v)]

            # -------------------------------------------------
            # Compute probabilistic vertex conflict
            # -------------------------------------------------
            p_cv = self.compute_p_conflict_cv(
                t_diff=abs(t_i - t_j),
                n1=n1,
                n2=n2,
                lam=lam,
                nv=nv
            )

            # print(path_i, '\n', path_j)
            # print(n1, n2, lam, nv)
            # print(p_cv, idx_i)

            if p_cv >= self.p_conflict_threshold:
                if self.debug:
                    print("ada konflik di", path_i[idx_i], "agen", agent_i, "pada timestep", idx_i, "dan agen", agent_j, 'pada timestep', idx_j)
                return idx_i, idx_j

        return -1, -1

    @staticmethod
    def dist(point1: np.ndarray, point2: np.ndarray) -> int:
        return int(np.linalg.norm(point1-point2, 2))  # L2 norm

    def calculate_constraints(self, node: CTNode,
                              constrained_agent: Agent,
                              unchanged_agent: Agent,
                              constrained_time: int,
                              unchanged_time: int) -> Constraints:
        constrained_path = node.solution[constrained_agent]
        unchanged_path = node.solution[unchanged_agent]

        pivot = unchanged_path[unchanged_time]
        conflict_end_time = constrained_time + 1

        while conflict_end_time < len(constrained_path):
            # nominal time difference = t_constrained - t_unchanged
            t_diff = float(conflict_end_time - unchanged_time)

            n1 = sum(
                self.st_planner.grid.n_v[tuple(constrained_path[k])]
                for k in range(conflict_end_time + 1)
            )

            n2 = sum(
                self.st_planner.grid.n_v[tuple(unchanged_path[k])]
                for k in range(unchanged_time + 1)
            )

            lam = self.st_planner.grid.lambda_v[tuple(pivot)]
            nv = self.st_planner.grid.n_v[tuple(pivot)]

            p_cv = self.compute_p_conflict_cv(
                t_diff=t_diff,
                n1=n1,
                n2=n2,
                lam=lam,
                nv=nv
            )

            if p_cv < self.p_conflict_threshold:
                break

            conflict_end_time += 1

        if self.debug:
            print(constrained_agent, tuple(pivot.tolist()), constrained_time, conflict_end_time)
        return node.constraints.fork(
            constrained_agent,
            tuple(pivot.tolist()),
            constrained_time,
            conflict_end_time
        )
    def calculate_goal_times(self, node: CTNode, agent: Agent, agents: List[Agent]):
        solution = node.solution
        goal_times = dict()
        for other_agent in agents:
            if other_agent == agent:
                continue
            time = len(solution[other_agent]) - 1
            goal_times.setdefault(time, set()).add(tuple(solution[other_agent][time]))
        return goal_times

    '''
    Calculate the paths for all agents with space-time constraints
    '''
    def calculate_path(self, agent: Agent, 
                       constraints: Constraints, 
                       goal_times: Dict[int, Set[Tuple[int, int]]]) -> np.ndarray:
        if self.debug:
            print("Constraints: ", constraints)
        return self.st_planner.plan(agent.start, 
                                    agent.goal, 
                                    constraints.setdefault(agent, dict()),
                                    # semi_dynamic_obstacles=goal_times,
                                    max_iter=self.low_level_max_iter, 
                                    debug=self.debug)

    '''
    Reformat the solution to a numpy array
    '''
    @staticmethod
    def reformat(agents: List[Agent], solution: Dict[Agent, np.ndarray]):
        solution = Planner.pad(solution)
        reformatted_solution = []
        for agent in agents:
            reformatted_solution.append(solution[agent])
        return np.array(reformatted_solution)

    '''
    Pad paths to equal length, inefficient but well..
    '''
    @staticmethod
    def pad(solution: Dict[Agent, np.ndarray]):
        max_ = max(len(path) for path in solution.values())
        for agent, path in solution.items():
            if len(path) == max_:
                continue
            padded = np.concatenate([path, np.array(list([path[-1]])*(max_-len(path)))])
            solution[agent] = padded
        return solution

    def compute_p_conflict_cv(self, t_diff, n1, n2, lam, nv, samples=5000):
        """
        Estimate vertex conflict probability Pcv using Monte Carlo.

        t_diff = t1 - t2 (planned time difference)
        n1, n2 = Gamma shape parameters
        lam = rate parameter
        node_time = how long agent occupies node
        """

        # sample delays
        # print(n1, lam, samples)
        d1 = self.st_planner.grid.sample_vertex_delays(shape=n1, rate=lam, samples=samples)
        d2 = self.st_planner.grid.sample_vertex_delays(shape=n2, rate=lam, samples=samples)

        tau_v1 = self.st_planner.grid.sample_vertex_delays(shape=nv, rate=lam, samples=samples)
        tau_v2 = self.st_planner.grid.sample_vertex_delays(shape=nv, rate=lam, samples=samples)

        # y = Δ1 - Δ2
        y = d1 - d2

        # actual time difference after delay
        shifted_diff = t_diff + (d1 - d2)

        # apply conditions (A) and (B)
        cond_A = shifted_diff + tau_v1 >= 0
        cond_B = -shifted_diff + tau_v2 >= 0

        collisions = cond_A & cond_B
        return np.mean(collisions)

    def compute_p_conflict_ce(t_diff, n1, n2, lam, te=1, samples=5000):
        """
        Estimate edge conflict probability Pce using Monte Carlo.

        t_diff = t1 - t2
        te = edge traversal time
        """

        d1 = np.random.gamma(shape=n1, scale=1 / lam, size=samples)
        d2 = np.random.gamma(shape=n2, scale=1 / lam, size=samples)

        # start times after delay
        a_start = d1
        b_start = d2 - t_diff  # align relative timing

        a_end = a_start + te
        b_end = b_start + te

        # check interval overlap
        collisions = ~((a_end < b_start) | (b_end < a_start))

        return np.mean(collisions)
