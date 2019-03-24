import torch
import numpy as np
import random

"""
World Info
- Small or big blob
- One or two small blobs
- One big blob
- Translation

Later, I will probably need to come up with a restriction on world transitions
so that everything is more intelligible. Intelligiblity only emerges by virtue
of a restriction on the possible states/transitions.
"""

show_blocks = [
        (0.9, '█'),
        (0.8, '▇'),
        (0.7, '▆'),
        (0.6, '▅'),
        (0.5, '▄'),
        (0.4, '▃'),
        (0.3, '▂'),
        (0.2, '▁'),
        (-.1, ' '),
        ]

def show(w):
    return "[" + "".join("█" if x > .9 else " " for x in w[0]) + "]"

def show_fancy(w):
    def to_block(x):
        for y, block in show_blocks:
            if x > y: return block
        return "X"
    # get the first channel of w
    return "[" + ''.join(to_block(x) for x in w[0]) + "]"



def make_world(size):
    makers = [
            lambda: blob(size, 6),
            lambda: blob(size, 3),
            lambda: blob_pair(size, 3),
            ]
    x = random.randint(0, len(makers) - 1)
    return makers[x]()

def blob(size, blob_size=6):
    w = torch.zeros(1, size)
    offset = random.randint(0, size - blob_size)
    for i in range(blob_size):
        w[0][offset + i] = 1.
    return w

def blob_pair(size, blob_size=3):
    w = torch.zeros(1, size)
    # total world size, space for 2 blobs, 1 space of separation
    offset = random.randint(0, size - 2 * blob_size - 1)
    for i in range(blob_size):
        w[0][offset + i] = 1.
    # separation accounted for in the 1
    offset_2 = random.randint(1, size - offset - 2*blob_size)
    for i in range(blob_size):
        w[0][offset + blob_size + offset_2 + i] = 1.
    return w
