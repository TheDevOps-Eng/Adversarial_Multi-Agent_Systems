
import numpy as np
import torch
from torch import nn

from tag.arena import MOVES, WALLS


class PursuerMLP(nn.Module):

    def __init__(self, observation_shape, hidden_size=256):
        super().__init__()

        input_size = int(np.prod(observation_shape))

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(MOVES))
        )

    def forward(self, x):
        return self.network(x.float())


def legal_action_mask(obs):

    radius = obs.shape[1] // 2
    mask = []

    for move in MOVES:
        r = radius + int(move[0])
        c = radius + int(move[1])

        legal = (
            0 <= r < obs.shape[1]
            and 0 <= c < obs.shape[2]
            and not obs[WALLS, r, c]
        )

        mask.append(legal)

    return np.array(mask, dtype=bool)


def load_policy(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    model = PursuerMLP(
        tuple(checkpoint["observation_shape"]),
        checkpoint["hidden_size"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["radius"]


def mlp_strategy(model):

    def choose_action(obs, rng):

        mask = legal_action_mask(obs)

        if not mask.any():
            return 0

        x = torch.tensor(
            obs[None, ...],
            dtype=torch.float32
        )

        with torch.no_grad():
            q_values = model(x)[0].numpy()

        q_values[~mask] = -np.inf

        return int(np.argmax(q_values))

    return choose_action
