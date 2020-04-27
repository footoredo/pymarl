import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import Attention


class ATTRNNAgentCompatible(nn.Module):
    def __init__(self, input_scheme, args):
        super(ATTRNNAgentCompatible, self).__init__()
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
            # print(var_inputs[i][1])
            vfcs.append(nn.Linear(var_inputs[i][1], args.attn_hidden_dim))
            len_attn += args.attn_hidden_dim * args.attn_n_heads
        ffc = nn.Linear(len_fixed, args.attn_hidden_dim * args.attn_n_heads)
        len_attn += args.attn_hidden_dim * args.attn_n_heads

        self.split = split
        self.input_scheme = input_scheme
        self.attns = nn.ModuleList(attns)
        self.vfcs = nn.ModuleList(vfcs)
        self.ffc = ffc

        self.fc1 = nn.Linear(len_attn, args.rnn_hidden_dim)
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
        split_inputs = inputs.split(self.split, dim=1)
        # print(" split_inputs[0]", split_inputs[0].is_cuda)
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
            # print("var_input", var_input.is_cuda)
            values = self.vfcs[i](var_input)
            attn_output = self.attns[i](fixed_input, var_input, values)
            var_outputs.append(attn_output)

        fixed_output = self.ffc(fixed_input)
        # print(fixed_output.size(), var_outputs[0].size())
        attn_output = th.cat([fixed_output] + var_outputs, dim=1)

        x = F.relu(self.fc1(attn_output))
        if self.args.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = x
        q = self.fc2(h)
        return q, h
