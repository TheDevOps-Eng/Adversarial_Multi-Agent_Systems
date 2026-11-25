
import argparse

import pygame

from tag.arena import EVADER, PURSUER, Arena
from tag.render import CELL, draw
from tag.mlp import load_policy, mlp_strategy
from tag.strategies import EVADER_STRATEGIES, PURSUER_STRATEGIES


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-ne", "--evaders", type=int, default=1)
    parser.add_argument("-np", "--pursuers", type=int, default=3)

    parser.add_argument(
        "-e",
        "--evader",
        choices=EVADER_STRATEGIES,
        default="random"
    )

    parser.add_argument(
        "-p",
        "--pursuer",
        choices=[*PURSUER_STRATEGIES, "mlp"],
        default="random"
    )

    parser.add_argument(
        "--model",
        default="models/pursuer_mlp.pt"
    )

    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-f", "--fps", type=int, default=8)
    parser.add_argument("-r", "--radius", type=int)

    args = parser.parse_args()

    arena = Arena(
        args.evaders,
        args.pursuers,
        args.seed
    )

    if args.pursuer == "mlp":

        model, trained_radius = load_policy(args.model)

        if args.radius is not None and args.radius != trained_radius:
            parser.error(
                f"Model requires radius {trained_radius}"
            )

        args.radius = trained_radius

        pursuer_policy = mlp_strategy(model)

    else:

        pursuer_policy = PURSUER_STRATEGIES[args.pursuer]

    strategies = {
        EVADER: EVADER_STRATEGIES[args.evader],
        PURSUER: pursuer_policy
    }

    pygame.init()

    h, w = arena.walls.shape

    screen = pygame.display.set_mode(
        (w * CELL, h * CELL)
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        if not arena.caught():

            actions = [
                strategies[role](
                    arena.observe(i, args.radius),
                    arena.rng
                )
                for i, role in enumerate(arena.roles)
            ]

            arena.step(actions)

        draw(screen, arena, args.radius)

        screen.blit(
            font.render(
                f"{args.evader} vs {args.pursuer}",
                True,
                "black",
                "white"
            ),
            (4, 4)
        )

        pygame.display.flip()

        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
