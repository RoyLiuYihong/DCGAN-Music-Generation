import pickle
import random
import numpy as np

class Dataset:
    def __init__(self, path):
        self.window_size = 64
        self.samples = pickle.load(open(path, 'rb'))
        self.indeces = []
        for i, sample in enumerate(self.samples):
            for j in range(len(sample) // self.window_size):
                self.indeces.append((i, j))

    def batches(self, n=64):
        batch = []
        while len(batch) < n:
            i, j = random.choice(self.indeces)
            x = self.samples[i][j:j+self.window_size]
            batch.append(x)
            if len(batch) == n:
                yield np.stack(batch).astype(np.float32)
                batch.clear()

