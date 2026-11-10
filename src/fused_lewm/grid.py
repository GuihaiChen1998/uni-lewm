"""A discrete two-room toy task, explicitly not the official TwoRoom benchmark."""

from collections import deque

import numpy as np
from PIL import Image


class TwoRoomGrid:
    num_actions = 4
    moves = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def __init__(self, size: int = 9, image_size: int = 32, seed: int = 0):
        if size < 7 or size % 2 == 0 or image_size < size:
            raise ValueError("use an odd grid size >=7 and image_size>=grid size")
        self.size, self.image_size = size, image_size
        self.rng = np.random.default_rng(seed)
        self.walls = np.zeros((size, size), dtype=bool)
        self.walls[[0, -1], :] = True
        self.walls[:, [0, -1]] = True
        self.walls[:, size // 2] = True
        self.walls[size // 2, size // 2] = False
        self.cells = [(int(x), int(y)) for y, x in np.argwhere(~self.walls)]
        self.position = self.cells[0]

    def reset(self, position=None):
        self.position = (
            tuple(position)
            if position is not None
            else self.cells[int(self.rng.integers(len(self.cells)))]
        )
        if self.position not in self.cells:
            raise ValueError("position must be a traversable cell")
        return self.render()

    def transition(self, position, action):
        if not 0 <= int(action) < self.num_actions:
            raise ValueError("STOP is handled by the controller, not by environment dynamics")
        x, y = position
        dx, dy = self.moves[int(action)]
        return (x + dx, y + dy) if not self.walls[y + dy, x + dx] else (x, y)

    def step(self, action):
        self.position = self.transition(self.position, action)
        return self.render()

    def render(self, position=None):
        x, y = self.position if position is None else position
        pixels = np.full((self.size, self.size, 3), (235, 239, 244), dtype=np.uint8)
        pixels[self.walls] = (37, 51, 77)
        pixels[y, x] = (241, 102, 53)
        resized = Image.fromarray(pixels).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        return np.asarray(resized).transpose(2, 0, 1).copy()

    def shortest_distance(self, start, goal):
        queue = deque([(tuple(start), 0)])
        seen = {tuple(start)}
        while queue:
            position, distance = queue.popleft()
            if position == tuple(goal):
                return distance
            for action in range(self.num_actions):
                nxt = self.transition(position, action)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        raise ValueError("unreachable goal")
