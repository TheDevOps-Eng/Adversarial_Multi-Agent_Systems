
import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from tag.arena import EVADER, PURSUER, Arena
from tag.mlp import PursuerMLP, legal_action_mask
from tag.strategies import EVADER_STRATEGIES


def select_action(model, obs, epsilon, rng, device):

    legal = legal_action_mask(obs)
    actions = np.flatnonzero(legal)

    if len(actions) == 0:
        return 0

    if rng.random() < epsilon:
        return int(rng.choice(actions))

    state = torch.tensor(
        obs[None, ...],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        q_values = model(state)[0].cpu().numpy()

    q_values[~legal] = -np.inf

    return int(np.argmax(q_values))


def nearest_distance(arena, agent_id):

    evaders = arena.evaders

    return int(
        np.abs(evaders - arena.positions[agent_id])
        .sum(axis=1).min()
    )


def train(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    observation_shape = (
        3,
        2 * args.radius + 1,
        2 * args.radius + 1
    )

    online = PursuerMLP(
        observation_shape,
        args.hidden_size
    ).to(device)

    target = PursuerMLP(
        observation_shape,
        args.hidden_size
    ).to(device)

    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(
        online.parameters(),
        lr=args.learning_rate
    )

    replay = deque(maxlen=args.replay_size)

    loss_function = nn.SmoothL1Loss()

    global_step = 0

    recent_capture = deque(maxlen=100)
    recent_return = deque(maxlen=100)

    evader_names = [
        name.strip()
        for name in args.evader_mix.split(",")
        if name.strip()
    ]

    if not evader_names:
        raise ValueError("At least one evader strategy is required.")

    for name in evader_names:
        if name not in EVADER_STRATEGIES:
            raise ValueError(f"Unknown evader strategy: {name}")

    for episode in range(1, args.episodes + 1):

        arena = Arena(
            args.evaders,
            args.pursuers,
            int(rng.integers(0, 2**31 - 1))
        )

        # Randomly select an evader strategy for each episode.
        evader_name = str(rng.choice(evader_names))
        evader_policy = EVADER_STRATEGIES[evader_name]

        episode_return = 0.0
        was_caught = False

        epsilon = args.epsilon_start

        for step in range(args.max_steps):

            pursuer_ids = [
                i for i, role in enumerate(arena.roles)
                if role == PURSUER
            ]

            observations = {
                i: arena.observe(i, args.radius)
                for i in pursuer_ids
            }

            old_distances = {
                i: nearest_distance(arena, i)
                for i in pursuer_ids
            }

            epsilon_fraction = min(
                1.0,
                global_step / max(1, args.epsilon_decay_steps)
            )

            epsilon = (
                args.epsilon_start
                + epsilon_fraction
                * (args.epsilon_end - args.epsilon_start)
            )

            actions = []

            for i, role in enumerate(arena.roles):

                obs = arena.observe(i, args.radius)

                if role == EVADER:

                    action = evader_policy(obs, rng)

                else:

                    action = select_action(
                        online,
                        obs,
                        epsilon,
                        rng,
                        device
                    )

                actions.append(action)

            arena.step(actions)

            was_caught = arena.caught()

            for agent_id in pursuer_ids:

                next_obs = arena.observe(
                    agent_id,
                    args.radius
                )

                new_distance = nearest_distance(
                    arena,
                    agent_id
                )

                reward = (
                    args.distance_reward
                    * (old_distances[agent_id] - new_distance)
                    - args.step_penalty
                )

                if was_caught:
                    reward += args.capture_reward

                replay.append((
                    observations[agent_id].astype(np.float32),
                    actions[agent_id],
                    float(reward),
                    next_obs.astype(np.float32),
                    was_caught
                ))

                episode_return += reward

            global_step += 1

            if (
                len(replay) >= args.batch_size
                and global_step % args.train_every == 0
            ):

                batch = random.sample(
                    replay,
                    args.batch_size
                )

                states, acts, rewards, next_states, dones = zip(*batch)

                states_t = torch.tensor(
                    np.stack(states),
                    dtype=torch.float32,
                    device=device
                )

                actions_t = torch.tensor(
                    acts,
                    dtype=torch.long,
                    device=device
                )

                rewards_t = torch.tensor(
                    rewards,
                    dtype=torch.float32,
                    device=device
                )

                next_states_t = torch.tensor(
                    np.stack(next_states),
                    dtype=torch.float32,
                    device=device
                )

                dones_t = torch.tensor(
                    dones,
                    dtype=torch.float32,
                    device=device
                )

                current_q = online(states_t).gather(
                    1,
                    actions_t.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():

                    next_q_values = target(next_states_t)

                    next_masks = np.stack([
                        legal_action_mask(obs)
                        for obs in next_states
                    ])

                    next_masks_t = torch.tensor(
                        next_masks,
                        dtype=torch.bool,
                        device=device
                    )

                    next_q_values = next_q_values.masked_fill(
                        ~next_masks_t,
                        -1e9
                    )

                    next_q = next_q_values.max(dim=1).values

                    expected_q = (
                        rewards_t
                        + args.gamma * (1 - dones_t) * next_q
                    )

                loss = loss_function(
                    current_q,
                    expected_q
                )

                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    online.parameters(),
                    args.gradient_clip
                )

                optimizer.step()

                if global_step % args.target_update == 0:

                    target.load_state_dict(
                        online.state_dict()
                    )

            if was_caught:
                break

        recent_capture.append(int(was_caught))
        recent_return.append(episode_return)

        if episode % args.log_every == 0 or episode == 1:

            print(
                f"Episode: {episode}/{args.episodes} | "
                f"Evader: {evader_name} | "
                f"Capture Rate: {np.mean(recent_capture):.2%} | "
                f"Return: {np.mean(recent_return):.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": online.cpu().state_dict(),
            "observation_shape": observation_shape,
            "radius": args.radius,
            "hidden_size": args.hidden_size,
            "episodes": args.episodes,
            "evader_mix": evader_names
        },
        output
    )

    print(f"Model saved at: {output}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--radius", type=int, default=6)

    parser.add_argument("--evaders", type=int, default=1)
    parser.add_argument("--pursuers", type=int, default=3)

    parser.add_argument(
        "--evader-mix",
        default="random,greedy"
    )

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100000)

    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=1000)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000)

    parser.add_argument("--capture-reward", type=float, default=100.0)
    parser.add_argument("--distance-reward", type=float, default=1.5)
    parser.add_argument("--step-penalty", type=float, default=0.05)

    parser.add_argument("--gradient-clip", type=float, default=10.0)

    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument(
        "--output",
        default="models/pursuer_mlp.pt"
    )

    train(parser.parse_args())


if __name__ == "__main__":
    main()
