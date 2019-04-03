import torch
import numpy as np
import random

"""
World transitions
- big blob split/merge
- single blob shrink-grow
- translate left/right
"""

class WorldPair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def to_tensor():
        return torch.stack((self.first, self.second))

class World1d:
    def __init__(self, length, depth):
        self.length = length
        self.depth = depth
        self.blobs = []

    def to_tensor(self):
        # TODO Lazy load this
        t = torch.zeros(self.depth, self.length)
        channel = 0
        for x, sz in self.blobs:
            for i in range(sz):
                t[channel][i + x] = 1.
        return t

    def __str__(self):
        return show_fancy(self.to_tensor())

    def __hash__(self):
        return hash(str(self.to_tensor()))

    def __eq__(self, other):
        return torch.eq(self.data, other.data).all()

class BigBlob(World1d):
    size_range = (6/24, 8.999/24)

    def __init__(self, length, depth, x, size):
        super(BigBlob, self).__init__(length, depth)
        #x = max(min(1., x), 0.)
        lo, hi = BigBlob.size_range
        self.size = int((lo + (hi - lo) * size) * self.length)
        right = (self.length - self.size)
        self.x = int(right * x)
        print(self.x)
        print(self.size)
        self.blobs.append((self.x, self.size))

def make_worlds(w_length):
    d = {
            'big_blob': [],
            'small_blob': [],
            'near_blobs': [],
            'far_blobs': [],
            }
    # TODO If this gets too big, generate via random index ordering
    for size in (6, 7, 8):
        for x in range(0, w_length - size + 1):
            w = World1d(w_length, 1)
            w.blobs.append((x, size))
            d['big_blob'].append(w)
    for size in (3, 4):
        for x in range(0, w_length - size + 1):
            w = World1d(w_length, 1)
            w.blobs.append((x, size))
            d['small_blob'].append(w)

    
    for size1 in (3, 4):
        for size2 in (3, 4):
            for sep in (2, 3):
                for x in range(0, w_length - (size1 + size2 + sep) + 1):
                    w = World1d(w_length, 1)
                    w.blobs.append((x, size1))
                    w.blobs.append((x+size1+sep, size2))
                    d['near_blobs'].append(w)
    for size1 in (3, 4):
        for size2 in (3, 4):
            for sep in (5, 6):
                for x in range(0, w_length - (size1 + size2 + sep) + 1):
                    w = World1d(w_length, 1)
                    w.blobs.append((x, size1))
                    w.blobs.append((x+size1+sep, size2))
                    d['far_blobs'].append(w)

    # ? Do the selection here or elsewhere?
    return d
    

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
