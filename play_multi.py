import time
from collections import deque

import pybullet as pb

from hivemind_env.env import HiveMindMultiAgentEnv


def resource_cells(env):
    return [
        env._world_to_grid(*pb.getBasePositionAndOrientation(resource_id, physicsClientId=env.client_id)[0][:2])
        for resource_id in env.resource_ids
    ]


def shelf_cells(cells, grid_size):
    resources = set(cells)
    return {
        (row, column)
        for row in range(1, grid_size - 1, 2)
        for column in range(1, grid_size - 1)
        if (row, column) not in resources
    }


def path_between(start, goal, blocked, grid_size):
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = [current]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return list(reversed(path))
        row, column = current
        for candidate in ((row - 1, column), (row + 1, column),
                          (row, column - 1), (row, column + 1)):
            if not (0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size):
                continue
            if candidate in blocked or candidate in previous:
                continue
            previous[candidate] = current
            queue.append(candidate)
    raise RuntimeError(f"No clear path from {start} to {goal}")


def direction_for(start, end):
    return {
        (0, 1): 0,
        (-1, 0): 1,
        (0, -1): 2,
        (1, 0): 3,
    }[(end[0] - start[0], end[1] - start[1])]


def approach_cell(resource, blocked, grid_size):
    candidates = [
        (resource[0] - 1, resource[1]),
        (resource[0] + 1, resource[1]),
        (resource[0], resource[1] - 1),
        (resource[0], resource[1] + 1),
    ]
    for candidate in candidates:
        if (0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size
                and candidate not in blocked and candidate != (0, 0)):
            return candidate
    raise RuntimeError(f"No clear approach cell for resource {resource}")


def depot_approach_cell(current, blocked, grid_size):
    candidates = [(0, 1), (1, 0)]
    candidates.sort(key=lambda cell: abs(cell[0] - current[0]) + abs(cell[1] - current[1]))
    for candidate in candidates:
        if candidate not in blocked and candidate != (0, 0):
            return candidate
    raise RuntimeError("No clear cell adjacent to the depot")


def turn_to(env, current_direction, target_direction):
    for _ in range((target_direction - current_direction) % 4):
        env.step([2, 6, 6, 6])
        current_direction = (current_direction + 1) % 4
    return current_direction


def navigate_to_cell(env, current, target, direction, blocked):
    for next_cell in path_between(current, target, blocked, env.grid_size)[1:]:
        direction = turn_to(env, direction, direction_for(current, next_cell))
        env.step([0, 6, 6, 6])
        current = next_cell
    return current, direction


def navigate_to_pickup_and_depot(env):
    cells = resource_cells(env)
    if not cells:
        raise RuntimeError("The reset world contains no resources")

    grid_size = env.grid_size
    resource = cells[0]
    blocked = shelf_cells(cells, grid_size) | {(0, 0)}
    approach = approach_cell(resource, blocked, grid_size)
    current = env._world_to_grid(*pb.getBasePositionAndOrientation(
        env.robot_ids[0], physicsClientId=env.client_id)[0][:2])
    direction = 0

    current, direction = navigate_to_cell(env, current, approach, direction, blocked)

    direction = turn_to(env, direction, direction_for(approach, resource))
    env.step([4, 6, 6, 6])
    if not env.is_carrying[0]:
        raise RuntimeError(f"Bot 0 failed to pick up resource at {resource}")
    print(f"Bot 0 picked up resource at {resource}; lidar raised to {env.lidar_carry_height} m")

    depot_approach = depot_approach_cell(current, blocked, grid_size)
    current, direction = navigate_to_cell(env, current, depot_approach, direction, blocked)
    direction = turn_to(env, direction, direction_for(depot_approach, (0, 0)))
    env.step([5, 6, 6, 6])
    if env.is_carrying[0]:
        raise RuntimeError("Bot 0 failed to drop the resource at the depot")
    print(f"Bot 0 dropped the resource from depot approach cell {depot_approach}")


def play_demo():
    print("Initializing single-bot resource pickup demo...")
    env = HiveMindMultiAgentEnv(render_mode="human")
    try:
        env.reset()
        pb.resetDebugVisualizerCamera(cameraDistance=16.0, cameraYaw=0,
                                      cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
        navigate_to_pickup_and_depot(env)
        time.sleep(2)
    except KeyboardInterrupt:
        print("Demo stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    play_demo()
