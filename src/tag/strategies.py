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

def genetic_strategy(arena: Arena, agent: int, opponents: np.ndarray, sign:int,) ->int:
    population_size = 15
    generations = 10
    chromosome_length = 5
    mutation_rate = 0.1

    rng = arena.rng

    if len(opponents) == 0:
        return STAY

    population = rng.integers(
        0,
        len(MOVES),
        size=(population_size, chromosome_length)
    )


    def fitness(chromosome):

        position = arena.positions[agent].copy()
        total_score = 0

        for action in chromosome:

            new_position = position + MOVES[action]

            if arena.is_free(*new_position):
                position = new_position

            distance = nearest_distance(position, opponents)

            total_score += sign * distance

        return total_score


    for generation in range(generations):

        scores = [fitness(chromosome) for chromosome in population]

        new_population = []

      
        best_index = int(np.argmax(scores))
        new_population.append(population[best_index].copy())

  
        while len(new_population) < population_size:


            def select_parent():

                candidates = rng.choice(
                    population_size,
                    size=3,
                    replace=False
                )

                winner = max(candidates, key=lambda i: scores[i])

                return population[winner]

            parent1 = select_parent()
            parent2 = select_parent()


            if chromosome_length > 1:
                point = int(rng.integers(1, chromosome_length))
                child = np.concatenate((
                    parent1[:point],
                    parent2[point:]
                ))
            else:
                child = parent1.copy()


            for i in range(chromosome_length):

                if rng.random() < mutation_rate:
                    child[i] = rng.integers(0, len(MOVES))

            new_population.append(child)

        population = np.array(new_population)

 
    final_scores = [fitness(chromosome) for chromosome in population]

    best_index = int(np.argmax(final_scores))

    best_chromosome = population[best_index]

    return int(best_chromosome[0])
    #pass

def genetic_evader(arena: Arena, agent: int) -> int:
    return genetic_strategy(
        arena,
        agent,
        arena.pursuers,
        1
    )


def genetic_pursuer(arena: Arena, agent: int) -> int:
    return genetic_strategy(
        arena,
        agent,
        arena.evaders,
        -1
    )
EVADER_STRATEGIES = {"random": random_strategy, "greedy": greedy_evader,"genetic":genetic_evader}
PURSUER_STRATEGIES = {"random": random_strategy, "greedy": greedy_pursuer, "astar": astar_pursuer,"genetic":genetic_pursuer}
