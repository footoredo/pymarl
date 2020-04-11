import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, key_dim, n_heads=1):
        super(Attention, self).__init__()

        self.query_input_dim = query_input_dim
        self.key_input_dim = key_input_dim
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.fcq = nn.ModuleList([nn.Linear(query_input_dim, key_dim) for _ in range(n_heads)])
        self.fck = nn.ModuleList([nn.Linear(key_input_dim, key_dim) for _ in range(n_heads)])

    def forward(self, queries, keys, values):
        """
            queries: batch * query_input_dim
            keys: batch * ? * key_input_dim
            values: batch * ? * value_dim
        """
        attns = []
        for h in range(self.n_heads):
            q = self.fcq[h](queries)  # batch * key_dim
            k = self.fck[h](keys)  # batch * ? * key_dim
            # print("q", q.size())
            # print("k", k.size())
            weight = th.bmm(q.view(-1, 1, self.key_dim), k.transpose(1, 2)) / np.sqrt(self.key_dim)  # batch * 1 * ?
            # print("weight", weight.size())
            attn = th.bmm(F.softmax(weight, dim=2), values).squeeze(1)  # batch * value_dim
            # print("attn", attn.size())
            attns.append(attn)
        return th.cat(attns, dim=1)
