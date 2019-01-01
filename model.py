import numpy as np
import torch
import torch.nn as nn

input_data = [[0,1,2],[1,2,3]]

class embedding(nn.Module):
    def __init__(self, voca_size, dimension):
        super(embedding, self).__init__()
        self.embed = nn.Embedding(voca_size, dimension)

def positional_encoding(x):
    dimension = x.shape[-1]
    position = x.shape[-2]
    batches = x.shape[-3]
    pe = np.array([[[pos / 10000**(2*i) for i in range(dimension)] for pos in range(position)] for batch in range(batches)])
    pe[:, :, 0::2] = np.sin(pe[:, :, 0::2])
    pe[:, :, 1::2] = np.cos(pe[:, :, 1::2])
    return torch.as_tensor(torch.from_numpy(pe), dtype=torch.float32) + x
