import numpy as np


class DataGenerator:
    def __init__(self, inputs, shuffle=True, batch_size=32):
        assert len(inputs) > 0
        self.inputs = inputs
        self.idx = np.arange(len(inputs[0]))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def data_length(self):
        return len(self.idx)

    def __len__(self):
        n = self.data_length()
        len_ = n // self.batch_size
        return len_ if n % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        index = self.idx[start:end]
        data = []
        for x in self.inputs:
            data.append(x[start:end])
        return data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
