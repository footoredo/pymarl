import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, key_dim, n_heads=1):
        super(Attention, self).__init__()

        self.query_input_dim = query_input_dim
        self.key_input_dim = key_input_dim
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.fcq = [nn.Linear(query_input_dim, key_dim) for _ in range(n_heads)]
        self.fck = [nn.Linear(key_input_dim, key_dim) for _ in range(n_heads)]

    def forward(self, queries, keys, values):
        """
            queries: batch * query_input_dim
            keys: batch * ? * key_input_dim
            values: batch * ? * value_dim
        """
        attns = []
        for h in range(self.n_heads):
            q = self.fcq[h](queries)  # batch * key_dim
            k = self.fcc[h](keys)  # batch * ? * key_dim
            weight = th.bmm(q.view(-1, 1, self.key_dim), k.transpose(1, 2)) / th.sqrt(self.key_dim)  # batch * 1 * ?
            attn = th.bmm(F.softmax(weight, dim=2), values).unsqueeze(1)  # batch * value_dim
            attns.append(attn)
        return th.cat(attns, dim=1)
