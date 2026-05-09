import json
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from gym.spaces import Dict as GymDict, Box, Discrete
from stt_mapf.grid import Grid as MAPF
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from collections import deque
import numpy as np
import random

# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    "custom": {
        "description": "MAPF cooperative agents",
        "team_prefix": ("agent_",),  # must match your agent names
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    }
}

class RLlibMAPF(MultiAgentEnv):

    def __init__(self, env_config):
        # pass YAML filename directly
        # scenario = r"C:\Users\Lenovo\Documents\PyCharm\mapf\stt_mapf\original.yaml"
        scenario = r"C:\Users\Lenovo\Documents\PyCharm\mapf\marllib\envs\base_env\config\mapf.yaml"

        # ---- BASE ENV ----
        self.env = MAPF(scenario)

        self.static_obstacles = {
            (int(o[0]), int(o[1])) for o in self.env.static_obstacles
        }

        # ---- AGENTS ----
        self.start_coordinates = self.env.interface.starts
        self.goal_coordinates = self.env.interface.goals

        self.agent_positions = list(self.start_coordinates)
        self.agent_goals = list(self.goal_coordinates)

        self.num_agents = len(self.start_coordinates)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.reached_goal = [False for _ in range(self.num_agents)]
        self.agent_positions_after_action = [
            [list(coord)] for coord in self.start_coordinates
        ]

        # ---- SPACES ----
        self.action_space = Discrete(5)

        # ---- COST TRACER ----
        self.costmap = {}
        for i, goal in enumerate(self.agent_goals):
            dist_map = self._bfs_cost_to_goal(goal)
            self.costmap[i] = self._normalize_invert_costmap(dist_map)

        # (x, y, gx, gy, delay, tx, ty)
        self.observation_space = GymDict({
            "obs": Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([
                    self.env.maxx,
                    self.env.maxy,
                    self.env.maxx,
                    self.env.maxy,
                    10.0,   # delay cap
                    self.env.maxx,
                    self.env.maxy
                ], dtype=np.float32),
                dtype=np.float32
            )
        })

        self.max_action = 10
        self.action_count = [0 for _ in range(self.num_agents)]

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------
    def reset(self, eval_mode=False):
        if not eval_mode:
            # all valid free cells
            free_cells = []

            for x in range(self.env.minx, self.env.maxx + 1):
                for y in range(self.env.miny, self.env.maxy + 1):
                    if (x, y) not in self.static_obstacles:
                        free_cells.append((x, y))

            # need 2 positions per agent
            required_cells = self.num_agents * 2

            if len(free_cells) < required_cells:
                raise ValueError(
                    "Not enough free cells for randomized agents/goals."
                )

            # sample unique cells
            sampled = random.sample(free_cells, required_cells)

            # split into starts and goals
            self.start_coordinates = sampled[:self.num_agents]
            self.goal_coordinates = sampled[self.num_agents:]

        self.agent_positions = list(self.start_coordinates)
        self.agent_goals = list(self.goal_coordinates)

        self.remaining_delay = [0.0 for _ in range(self.num_agents)]
        self.target_positions = list(self.agent_positions)

        self.reached_goal = [False for _ in range(self.num_agents)]
        self.agent_positions_after_action = [
            [list(coord)] for coord in self.start_coordinates
        ]

        self.action_count = [0 for _ in range(self.num_agents)]

        return self._build_obs()

    # --------------------------------------------------
    # STEP (EVENT-DRIVEN)
    # --------------------------------------------------
    def step(self, action_dict):
        # ---- ASSIGN TARGETS (only free agents) ----
        for i, agent in enumerate(self.agents):

            if self.remaining_delay[i] <= 0 and self.action_count[i] < self.max_action:

                action = action_dict[agent]
                move = self._action_to_delta(action)

                current = self.agent_positions[i]
                target = self._move(current, move)

                self.target_positions[i] = target
                self.agent_positions_after_action[i].append(list(target))
                self.action_count[i] += 1

                delay = self.env.sample_vertex_delay(current)
                self.remaining_delay[i] = delay

        # ---- FIND NEXT EVENT ----
        active_delays = [
            self.remaining_delay[i]
            for i in range(self.num_agents)
            if self.action_count[i] < self.max_action
        ]

        if len(active_delays) > 0:
            dt = min(active_delays)
        else:
            dt = 0.0

        # ---- ADVANCE TIME ----
        for i in range(self.num_agents):
            self.remaining_delay[i] -= dt

        # ---- READY AGENTS ----
        ready = [i for i in range(self.num_agents) if self.remaining_delay[i] <= 0]

        prev_positions = list(self.agent_positions)
        next_positions = list(self.agent_positions)

        for i in ready:
            next_positions[i] = self.target_positions[i]

        # ---- CONFLICT CHECK ----
        conflicts = self._detect_conflicts(self.agent_positions, next_positions)

        # ---- RESOLVE CONFLICTS ----
        for (i, j) in conflicts:
            next_positions[i] = self.agent_positions[i]
            next_positions[j] = self.agent_positions[j]

        # ---- UPDATE STATE ----
        self.agent_positions = next_positions

        # ---- REWARD ----
        rewards = {}
        for i, agent in enumerate(self.agents):
            if any(i in c for c in conflicts):
                rewards[agent] = -5.0
            elif next_positions[i] == self.agent_goals[i] and prev_positions[i] != self.agent_goals[i] and not self.reached_goal[i]:
                rewards[agent] = 10.0
                self.reached_goal[i] = True
            elif next_positions[i] == self.agent_goals[i] and prev_positions[i] == self.agent_goals[i]:
                rewards[agent] = 0.0
            elif next_positions[i] != prev_positions[i]: # cost tracer
                local_costmap_obs = self._get_local_costmap_obs(
                    self.costmap[i],
                    pos=prev_positions[i],
                    target=next_positions[i]
                )

                if local_costmap_obs[0] < local_costmap_obs[1]:
                    rewards[agent] = 0.5
                else:
                    rewards[agent] = -0.5

            else:
                rewards[agent] = -0.3  # small time penalty

        # ---- DONE ----
        success = all(
            self.agent_positions[i] == self.agent_goals[i]
            for i in range(self.num_agents)
        )

        if success:
            for i, agent in enumerate(self.agents):
                rewards[agent] = 20.0

            # -------------------------------------------------
            # NORMALIZE PATH LENGTHS
            # -------------------------------------------------
            max_count = max(self.action_count) + 1
            for i in range(self.num_agents):
                current_len = len(self.agent_positions_after_action[i])
                if current_len < max_count:
                    last_pos = self.agent_positions_after_action[i][-1]
                    pad_amount = max_count - current_len

                    for _ in range(pad_amount):
                        self.agent_positions_after_action[i].append(
                            list(last_pos)
                        )

            # -------------------------------------------------
            # FIND FIRST GOAL INDICES
            # -------------------------------------------------
            first_goal_indices = []
            for i in range(self.num_agents):
                goal = list(self.goal_coordinates[i])
                path = self.agent_positions_after_action[i]
                goal_index = None

                for t, pos in enumerate(path):
                    if pos == goal:
                        goal_index = t
                        break
                first_goal_indices.append(goal_index)

            # -------------------------------------------------
            # GLOBAL MAX GOAL TIME
            # -------------------------------------------------

            max_goal_index = max(first_goal_indices)

            # -------------------------------------------------
            # TRIM / PAD TO SAME LENGTH
            # -------------------------------------------------

            for i in range(self.num_agents):
                path = self.agent_positions_after_action[i]
                goal_idx = first_goal_indices[i]

                # trim after first goal
                trimmed = path[:goal_idx + 1]

                # pad until global max goal time
                last_pos = trimmed[-1]

                while len(trimmed) < max_goal_index + 1:
                    trimmed.append(list(last_pos))

                self.agent_positions_after_action[i] = trimmed

        timeout = all(
            count >= self.max_action
            for count in self.action_count
        )

        done = success or timeout

        dones = {"__all__": done}

        # ---- OBS ----
        obs = self._build_obs()

        # ---- LOG ----
        with open("debug.log", "a") as f:
            f.write(f"action: {self.action_count}\n")
            f.write(f"remaining_delay: {self.remaining_delay}\n")
            f.write(f"OBS: {obs}\n")
            f.write(f"REWARDS: {rewards}\n")
            f.write(f"POSITIONS: {next_positions}")
            f.write(f"GOALS: {self.agent_goals}")
            f.write(f"ACTIONS: {action_dict}")
            f.write(f"CONFLICTS: {conflicts}")
            f.write(f"DONES: {dones}\n")
            f.write("=" * 40 + "\n")

        return obs, rewards, dones, {}

    # --------------------------------------------------
    # OBS
    # --------------------------------------------------
    def _build_obs(self):

        obs = {}

        for i, agent in enumerate(self.agents):

            pos = self.agent_positions[i]
            goal = self.agent_goals[i]
            target = self.target_positions[i]
            delay = max(self.remaining_delay[i], 0.0)

            obs[agent] = {
                "obs": np.array([
                    pos[0], pos[1],
                    goal[0], goal[1],
                    delay,
                    target[0], target[1]
                ], dtype=np.float32)
            }

        return obs

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _bfs_cost_to_goal(self, goal):
        """
        BFS shortest-path distance map using:
            self.env.minx
            self.env.maxx
            self.env.miny
            self.env.maxy

        Assumes:
            self.static_obstacles = set of (x, y)

        Returns:
            dict[(x, y)] = shortest distance to goal
        """

        # distance dictionary
        dist = {}

        # initialize all cells as inf
        for x in range(self.env.minx, self.env.maxx + 1):
            for y in range(self.env.miny, self.env.maxy + 1):
                dist[(x, y)] = np.inf

        gx, gy = goal

        # goal invalid
        if (gx, gy) in self.static_obstacles:
            return dist

        q = deque()
        q.append((gx, gy))

        dist[(gx, gy)] = 0

        directions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]

        while q:
            x, y = q.popleft()

            current_dist = dist[(x, y)]

            for dx, dy in directions:
                nx = x + dx
                ny = y + dy

                # boundary check
                if nx < self.env.minx or nx > self.env.maxx:
                    continue

                if ny < self.env.miny or ny > self.env.maxy:
                    continue

                # obstacle check
                if (nx, ny) in self.static_obstacles:
                    continue

                # already visited
                if dist[(nx, ny)] != np.inf:
                    continue

                dist[(nx, ny)] = current_dist + 1

                q.append((nx, ny))

        return dist

    def _normalize_invert_costmap(self, dist_map):
        """
        Converts BFS distances into:
            1 = closest to goal
            0 = far
           -1 = obstacle/unreachable
        """

        valid_distances = [
            d for d in dist_map.values()
            if np.isfinite(d)
        ]

        if len(valid_distances) == 0:
            return {}

        max_dist = max(valid_distances)

        result = {}

        for pos, d in dist_map.items():

            if not np.isfinite(d):
                result[pos] = -1.0

            else:
                normalized = d / max_dist
                inverted = 1.0 - normalized

                result[pos] = inverted

        return result

    def _get_local_costmap_obs(self, costmap, pos, target):
        """
        Returns:
            np.array([
                current_position_cost,
                target_position_cost
            ], dtype=np.float32)

        pos:
            current agent position (x, y)

        target:
            target position after movement (x, y)
        """

        ax, ay = pos
        tx, ty = target

        # current position cost
        current_cost = costmap.get((ax, ay), -1.0)

        # target position cost
        if (
                tx < self.env.minx or tx > self.env.maxx or
                ty < self.env.miny or ty > self.env.maxy
        ):
            target_cost = -1.0

        else:
            target_cost = costmap.get((tx, ty), -1.0)

        return np.array(
            [current_cost, target_cost],
            dtype=np.float32
        )
    def _action_to_delta(self, action):
        return {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
        }[action]

    def _move(self, pos, delta):
        x, y = pos
        dx, dy = delta

        nx, ny = x + dx, y + dy

        # boundary check
        if nx < self.env.minx or nx > self.env.maxx:
            return (x, y)
        if ny < self.env.miny or ny > self.env.maxy:
            return (x, y)

        # obstacle check
        if (nx, ny) in self.static_obstacles:
            return (x, y)

        return (nx, ny)

    def _detect_conflicts(self, old_pos, new_pos):

        conflicts = []

        # vertex conflict
        for i in range(len(new_pos)):
            for j in range(i + 1, len(new_pos)):
                if new_pos[i] == new_pos[j]:
                    conflicts.append((i, j))

        # edge conflict (swap)
        for i in range(len(new_pos)):
            for j in range(i + 1, len(new_pos)):
                if old_pos[i] == new_pos[j] and old_pos[j] == new_pos[i]:
                    conflicts.append((i, j))

        return conflicts
    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 100,
            "policy_mapping_info": policy_mapping_dict
        }

class Evaluate():
    def render_paths(env, model, restore_path, local_mode=False, num_workers=0, max_steps=100):
        # ---------------------------
        # LOAD CONFIG
        # ---------------------------
        with open(restore_path["params_path"], "r") as f:
            config = json.load(f)

        env_instance, env_info = env
        model_class, model_info = model

        num_agents = env_instance.num_agents

        # ---------------------------
        # CRITICAL: REGISTER MODEL AGAIN
        # ---------------------------
        ModelCatalog.register_custom_model(
            "Centralized_Critic_Model",
            model_class
        )

        # ---------------------------
        # FORCE RLLIB RUNTIME SETTINGS
        # ---------------------------
        config["num_workers"] = num_workers

        # ---------------------------
        # MULTIAGENT POLICY SETUP
        # ---------------------------
        config["multiagent"]["policies"] = {
            "shared_policy": (
                None,
                env_instance.observation_space,
                env_instance.action_space,
                {}
            )
        }

        # IMPORTANT: correct mapping for agent_0, agent_1, ...
        config["multiagent"]["policy_mapping_fn"] = (
            lambda agent_id: "shared_policy"
        )

        custom_config = config["model"]["custom_model_config"]

        # overwrite with real Gym spaces from env (GROUND TRUTH)
        custom_config["space_act"] = env_instance.action_space
        custom_config["space_obs"] = env_instance.observation_space

        # ---------------------------
        # INIT TRAINER
        # ---------------------------
        trainer = MAPPOTrainer(config=config)
        trainer.restore(restore_path["model_path"])

        # ---------------------------
        # RESET ENV
        # ---------------------------
        obs = env_instance.reset(eval_mode=True)
        done = False

        # ---------------------------
        # ROLLOUT LOOP
        # ---------------------------
        while not done and any(c < 100 for c in env_instance.action_count):
            # compute actions per agent
            action_dict = {}

            for agent_id, agent_obs in obs.items():

                policy_id = config["multiagent"]["policy_mapping_fn"](agent_id)
                action = trainer.compute_single_action(
                    agent_obs,
                    policy_id=policy_id
                )

                action_dict[agent_id] = action

            # step environment
            obs, rewards, dones, _ = env_instance.step(action_dict)
            done = dones["__all__"]

        paths = env_instance.agent_positions_after_action
        return np.array(paths)

if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["mapf"] = RLlibMAPF
    # initialize env
    env = marl.make_env(
        environment_name="mapf",
        map_name="custom"
    )
    # pick mappo algorithms
    # customize model
    mappo = marl.algos.mappo(hyperparam_source="mapf")
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
    # start learning
    # mappo.fit(
    #   env,
    #   model,
    #   checkpoint_freq=100,
    #   checkpoint_end=True,
    #   share_policy="all"
    # )

    paths = Evaluate.render_paths(
        env,
        model,
        restore_path={
            "params_path": r"C:\Users\Lenovo\Documents\PyCharm\mapf\marllib\examples\exp_results\mappo_mlp_custom\MAPPOTrainer_mapf_custom_ae4d1_00000_0_2026-05-08_16-17-05\params.json",
            "model_path": r"C:\Users\Lenovo\Documents\PyCharm\mapf\marllib\examples\exp_results\mappo_mlp_custom\MAPPOTrainer_mapf_custom_ae4d1_00000_0_2026-05-08_16-17-05\checkpoint_000100\checkpoint-100"
        },
        local_mode=True,
        num_workers=0
    )

    print(paths)