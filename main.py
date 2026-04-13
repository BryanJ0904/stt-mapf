from mapf_stochastic.grid import Grid as MAPF
from stt_cbs_mapf.planner import Planner as CBSPlanner
from stastar_stochastic.planner import Planner as AStarPlanner
# from visualization.visualizer import Simulator, load_scenario

if __name__ == "__main__":
    """
        This is how the single agent pathfinding works

        a_star_planner = AStarPlanner(grid_size=1, robot_radius=1, static_obstacles=[(0, 1), (11, 11)])
        assignment = a_star_planner.plan(start=[0, 3], goal=[9, 8], dynamic_obstacles=dynamic_obstacles)
        print(assignment)
    """

    mapf_env = MAPF(grid_size=1, robot_radius=1, static_obstacles=[(0, 0), (7, 4), (7, 7), (7, 3), (7, 6), (7, 5), (7, 8), (15, 15)], n_v=1.0, lambda_v=3.0)

    start_coordinates = [(5, 3), (4, 5), (9, 6)]
    goal_coordinates = [(5, 6), (6, 5), (4, 6)]

    cbs_planner = CBSPlanner(grid=mapf_env)

    print("Running STT-CBS...")
    assignment1 = cbs_planner.plan(
        starts=start_coordinates,
        goals=goal_coordinates,
        epsilon=0.001,
        debug=True
    )
    print("Result: ", assignment1)
    print("STT-CBS Done")

    # GRID_SIZE, ROBOT_RADIUS, RECT_OBSTACLES, START, GOAL = load_scenario('visualization/scenario1.yaml')
    # simulator = Simulator()
    # simulator.start()

    # a_star_planner = AStarPlanner(grid=mapf_env)
    # assignment = path = a_star_planner.plan(
    #     start=(0, 0),
    #     goal=(5, 5),
    #     dynamic_obstacles={
    #         3: {(2, 2)},
    #         4: {(2, 3)},
    #         5: {(2, 4)}
    #     },
    #     debug=True
    # )
    # print(assignment)