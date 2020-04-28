import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.multi_attention import MultiAttention


class MULTIATTRNNAgent(nn.Module):
    def __init__(self, input_scheme, args):
        super(MULTIATTRNNAgent, self).__init__()
        self.args = args

        n_layers = args.attn_n_layers
        n_heads = args.attn_n_heads
        hidden_dim = args.attn_hidden_dim

        self.attn = MultiAttention(input_scheme, n_layers, hidden_dim, n_heads)

        self.fc1 = nn.Linear(hidden_dim, args.rnn_hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        else:
            self.rnn = None
        # print(args.n_actions)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        attn_x = self.attn(inputs)

        x = F.relu(self.fc1(F.relu(attn_x)))
        if self.args.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = x
        q = self.fc2(h)
        return q, h
