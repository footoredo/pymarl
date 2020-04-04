import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import Attention


class ATTRNNAgent(nn.Module):
    def __init__(self, input_scheme, args):
        super(ATTRNNAgent, self).__init__()
        self.args = args

        fixed_inputs = []
        var_inputs = []
        idx = 0
        len_fixed = 0
        split = []
        for part in input_scheme:
            if type(part) == int:
                # part: len
                fixed_inputs.append((idx, part))
                idx += part
                len_fixed += part
                split.append(part)
            else:
                # part: len * n
                var_inputs.append((idx, part[0], part[1]))
                idx += part[0] * part[1]
                split.append(part[0] * part[1])

        attns = []
        vfcs = []
        n_var = len(var_inputs)
        len_attn = 0
        for i in range(n_var):
            attns.append(Attention(len_fixed, var_inputs[i][1], args.attn_hidden_dim, args.attn_n_heads))
            vfcs.append(nn.Linear(var_inputs[i][1], args.attn_hidden_dim))
            len_attn += args.attn_hidden_dim * args.attn_n_heads
        ffc = nn.Linear(len_fixed, args.attn_hidden_dim * args.attn_n_heads)
        len_attn += args.attn_hidden_dim * args.attn_n_heads

        self.split = split
        self.input_scheme = input_scheme
        self.attns = attns
        self.vfcs = vfcs
        self.ffc = ffc

        self.fc1 = nn.Linear(len_attn, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        split_inputs = inputs.split(self.split, dim=1)
        fixed_inputs = []
        var_inputs = []
        for i, part in enumerate(self.input_scheme):
            if type(part) == int:
                fixed_inputs.append(split_inputs[i])
            else:
                var_inputs.append(split_inputs[i].view(-1, part[1], part[0]))

        fixed_input = th.cat(fixed_inputs, dim=1)
        var_outputs = []
        for i, var_input in enumerate(var_inputs):
            values = self.vfcs[i](var_input)
            attn_output = self.attns[i](fixed_input, var_input, values)
            var_outputs.append(attn_output)

        fixed_output = self.ffc(fixed_input)
        attn_output = th.cat([fixed_output] + var_outputs, dim=1)

        x = F.relu(self.fc1(attn_output))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
