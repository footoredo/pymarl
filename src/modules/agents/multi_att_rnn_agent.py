import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.multi_attention import InputBuilder, MultiLayerAttention, ActionBuilder


class MULTIATTRNNAgent(nn.Module):
    def __init__(self, scheme, args):
        super(MULTIATTRNNAgent, self).__init__()
        self.args = args

        n_layers = args.attn_n_layers
        n_heads = args.attn_n_heads
        hidden_dim = args.attn_hidden_dim

        self.build_input = InputBuilder(scheme, hidden_dim)
        n_var = self.build_input.n_var

        self.attn = MultiLayerAttention(n_layers, n_var, hidden_dim, n_heads)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = None
        # print(args.n_actions)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if args.variable_action:
            self.build_action = ActionBuilder(scheme, hidden_dim, version=args.action_attention_version)
        else:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        fixed_state, var_states = self.build_input(inputs)
        list_var_states = [var_states[k] for k in sorted(var_states.keys())]
        attn_x = self.attn(fixed_state, list_var_states)

        x = F.relu(self.fc1(F.relu(attn_x)))
        if self.args.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = x
        if self.args.variable_action:
            q = self.build_action(h, var_states)
        else:
            q = self.fc2(h)
        return q, h
