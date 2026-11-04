import heapq

import numpy as np

from tag.arena import MOVES, STAY, Arena


def legal_actions(arena: Arena, agent: int) -> list[int]:
    return [a for a, move in enumerate(MOVES) if arena.is_free(*(arena.positions[agent] + move))]


def nearest_distance(cell: np.ndarray, others: np.ndarray) -> int:
    return int(np.abs(others - cell).sum(axis=1).min())


def random_strategy(arena: Arena, agent: int) -> int:
    """Choose one legal action at random, including staying in place."""
    actions = legal_actions(arena, agent)
    action = arena.rng.choice(actions)
    return int(action)


def greedy(arena: Arena, agent: int, opponents: np.ndarray, sign: int) -> int:
    actions = legal_actions(arena, agent)
    scores = [sign * nearest_distance(arena.positions[agent] + MOVES[a], opponents) for a in actions]
    best = [a for a, s in zip(actions, scores) if s == max(scores)]
    return int(arena.rng.choice(best))


def greedy_pursuer(arena: Arena, agent: int) -> int:
    return greedy(arena, agent, arena.evaders, -1)


def greedy_evader(arena: Arena, agent: int) -> int:
    return greedy(arena, agent, arena.pursuers, 1)


def astar_pursuer(arena: Arena, agent: int) -> int:
    """Find a shortest route to a cell next to an evader and return its first action."""
    if len(arena.evaders) == 0:
        return STAY
    start = tuple(arena.positions[agent])

    def heuristic(cell):
        # Capture happens at distance <= 1, so stop one cell before the evader.
        return max(0, nearest_distance(np.array(cell), arena.evaders) - 1)

    # Entries contain: estimated total cost, travelled cost, cell, first action.
    frontier = [(heuristic(start), 0, start, STAY)]
    costs = {start: 0}
    while frontier:
        _, cost, cell, first_action = heapq.heappop(frontier)
        if cost != costs[cell]:
            continue
        if heuristic(cell) == 0:
            return first_action
        for action, move in enumerate(MOVES):
            if action == STAY:
                continue
            next_cell = (cell[0] + move[0], cell[1] + move[1])
            if not arena.is_free(*next_cell):
                continue
            new_cost = cost + 1
            if new_cost < costs.get(next_cell, float("inf")):
                costs[next_cell] = new_cost
                first = action if cell == start else first_action
                priority = new_cost + heuristic(next_cell)
                heapq.heappush(frontier, (priority, new_cost, next_cell, first))
    return STAY  # No route to any evader.


EVADER_STRATEGIES = {"random": random_strategy, "greedy": greedy_evader}
PURSUER_STRATEGIES = {"random": random_strategy, "greedy": greedy_pursuer, "astar": astar_pursuer}
