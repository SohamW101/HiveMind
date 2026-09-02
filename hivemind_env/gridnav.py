"""
Grid helpers shared by the verification scripts: read the warehouse, BFS across it.

These lived in `play_multi.py`, a scripted one-robot demo that drove to a carton and
stopped. The demo was superseded by `hivemind_env/greedy.py`, which does the whole job
for four robots and is scored through the real evaluation harness - but two verifiers
still imported these five functions out of it, which is why every script carried
sys.path boilerplate to reach a module in the repo root. Moving them into the package
retired the demo and the boilerplate together.

The env's own `blocked_cells` is the authority on what is solid during a step; these
are for scripts that need to plan a route before stepping.
"""
from collections import deque

import pybullet as pb


def resource_cells(env):
    """Grid cells of every carton still on the floor, in the env's own order."""
    return [
        env._world_to_grid(*pb.getBasePositionAndOrientation(
            rid, physicsClientId=env.client_id)[0][:2])
        for rid in env.resource_ids
    ]


def shelf_cells(cells, grid_size):
    """
    Shelving, inferred from where the cartons are not.

    Shelf rows are the odd grid rows; a carton sits in each gap cut into them, so every
    odd-row cell that is not a carton is shelf. Inferred rather than read off the env
    because these run before the env exposes a blocked set.
    """
    resources = set(cells)
    return {
        (row, column)
        for row in range(1, grid_size - 1, 2)
        for column in range(1, grid_size - 1)
        if (row, column) not in resources
    }


def path_between(start, goal, blocked, grid_size):
    """Shortest 4-connected path, inclusive of both ends. Raises if there is none."""
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
    """Heading index (0-3) for a one-cell step. Raises on a non-adjacent pair."""
    return {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}[
        (end[0] - start[0], end[1] - start[1])
    ]


def approach_cell(resource, blocked, grid_size):
    """
    A free cell adjacent to `resource` to stand in while picking it up.

    The depot at (0, 0) is excluded: standing there to grab a carton would also put the
    robot in drop range, which makes a pickup/delivery test measure two things at once.
    """
    for candidate in ((resource[0] - 1, resource[1]), (resource[0] + 1, resource[1]),
                      (resource[0], resource[1] - 1), (resource[0], resource[1] + 1)):
        if (0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size
                and candidate not in blocked and candidate != (0, 0)):
            return candidate
    raise RuntimeError(f"No clear approach cell for resource {resource}")
