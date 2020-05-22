import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class InputBuilder(nn.Module):
    def __init__(self, input_scheme, hidden_dim):
        super(InputBuilder, self).__init__()

        fixed_inputs = []
        var_inputs = dict()
        n_var = 0
        len_fixed = 0
        split = []
        for part in input_scheme["observation_pattern"]:
            if type(part) == int:
                # part: len
                l = part
                fixed_inputs.append(len(split))
                len_fixed += l
                split.append(l)
            else:
                # part: len * name
                l, name = part
                n = input_scheme["objects"][name]
                if name not in var_inputs:
                    var_inputs[name] = {
                        "pos": [],
                        "len": 0,
                        "n": n
                    }
                    n_var += 1
                var_inputs[name]["pos"].append((len(split), l))
                var_inputs[name]["len"] += l
                split.append(l * n)

        fixed_fc = nn.Linear(len_fixed, hidden_dim)
        var_fcs = dict()
        for k, info in var_inputs.items():
            var_fcs[k] = nn.Linear(info["len"], hidden_dim)

        self.split = split
        # self.input_scheme = deepcopy(input_scheme)
        self.fixed_inputs_info = fixed_inputs
        self.var_inputs_info = var_inputs

        self.fixed_fc = fixed_fc
        self.var_fcs = nn.ModuleDict(var_fcs)
        self.n_var = n_var
        # self.keys = var_inputs.keys()

    def forward(self, inputs):
        split_inputs = inputs.split(self.split, dim=-1)
        # print(" split_inputs[0]", split_inputs[0].is_cuda)
        fixed_inputs = [split_inputs[idx] for idx in self.fixed_inputs_info]
        fixed_input = th.cat(fixed_inputs, dim=-1)
        fixed_state = self.fixed_fc(fixed_input)

        var_states = dict()
        for k, info in self.var_inputs_info.items():
            n = info["n"]
            parts = []
            for idx, l in info["pos"]:
                parts.append(split_inputs[idx].reshape(-1, n, l))
            var_input = th.cat(parts, dim=-1)
            var_states[k] = self.var_fcs[k](var_input)
        # print(self.input_scheme)
        # for i, part in enumerate(self.input_scheme):
        #     if type(part) == int:
        #         fixed_inputs.append(split_inputs[i])
        #     else:
        #         # print(part)
        #         # print(part, split_inputs[i].shape)
        #         var_inputs.append(split_inputs[i].view(-1, part[1], part[0]))

        # var_states = []
        # for i, var_input in enumerate(var_inputs):
        #     var_states.append(self.var_fcs[i](var_input))

        return fixed_state, var_states


class Attention(nn.Module):
    def __init__(self, model_dim, n_heads=1):
        super(Attention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // n_heads
        self.n_heads = n_heads
        self.fcq = nn.Linear(model_dim, self.dim_per_head * n_heads)
        self.fck = nn.Linear(model_dim, self.dim_per_head * n_heads)
        self.fcv = nn.Linear(model_dim, self.dim_per_head * n_heads)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, queries, keys, values):
        """
            queries: batch * model_dim
            keys: batch * ? * model_dim
            values: batch * ? * model_dim
        """
        residual = queries
        batch_size = queries.size(0)

        # (batch * n_heads) * (?) * dim_per_head
        q = self.fcq(queries).view(batch_size * self.n_heads, 1, self.dim_per_head)
        k = self.fck(keys).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2).\
            reshape(batch_size * self.n_heads, -1, self.dim_per_head)
        v = self.fcv(values).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2).\
            reshape(batch_size * self.n_heads, -1, self.dim_per_head)

        weight = th.bmm(q, k.transpose(1, 2)) / np.sqrt(self.dim_per_head)  # (batch * n_heads) * 1 * (?)
        attn = th.bmm(F.softmax(weight, dim=-1), v)  # (batch * n_heads) * 1 * dim_per_head
        attn = attn.view(batch_size, self.n_heads * self.dim_per_head)

        # for h in range(self.n_heads):
        #     q = self.fcq[h](queries)  # batch * dim_per_head
        #     k = self.fck[h](keys)  # batch * ? * dim_per_head
        #     v = self.fcv[h](values)
        #     # print("q", q.size())
        #     # print("k", k.size())
        #     weight = th.bmm(q.view(-1, 1, self.dim_per_head), k.transpose(1, 2)) / np.sqrt(self.dim_per_head)  # batch * 1 * ?
        #     # print("weight", weight.size())
        #     attn = th.bmm(F.softmax(weight, dim=2), v).squeeze(1)  # batch * value_dim
        #     # print("attn", attn.size())
        #     attns.append(attn)
        return self.layer_norm(attn + residual)


class AttentionLayer(nn.Module):
    def __init__(self, n_var, hidden_dim, n_heads):
        super(AttentionLayer, self).__init__()

        attns = []
        lns = []
        fc1s = []
        fc2s = []

        for i in range(n_var):
            attns.append(Attention(hidden_dim, n_heads))
            fc1s.append(nn.Linear(hidden_dim, hidden_dim))
            fc2s.append(nn.Linear(hidden_dim, hidden_dim))
            lns.append(nn.LayerNorm(hidden_dim))

        fc = nn.Linear(hidden_dim * n_var, hidden_dim)
        ln = nn.LayerNorm(hidden_dim)

        self.n_var = n_var

        self.attns = nn.ModuleList(attns)
        self.fc1s = nn.ModuleList(fc1s)
        self.fc2s = nn.ModuleList(fc2s)
        self.lns = nn.ModuleList(lns)
        self.fc = fc
        self.ln = ln

    def forward(self, query, states):
        residual = query

        xs = []
        for i in range(self.n_var):
            h = self.attns[i](query, states[i], states[i])
            x = self.fc2s[i](F.relu(self.fc1s[i](h)))
            x = self.lns[i](x + h)
            xs.append(x)

        x = th.cat(xs, dim=1)
        y = self.fc(x)
        y = self.ln(y + residual)
        return y


class MultiLayerAttention(nn.Module):
    def __init__(self, n_layers, n_var, hidden_dim, n_heads):
        super(MultiLayerAttention, self).__init__()

        attns = []
        for i in range(n_layers):
            attns.append(AttentionLayer(n_var, hidden_dim, n_heads))

        self.n_layers = n_layers
        self.attns = nn.ModuleList(attns)

    def forward(self, query, states):
        for i in range(self.n_layers):
            query = self.attns[i](query, states)
        return query


class MultiAttention(nn.Module):
    def __init__(self, input_scheme, n_layers, hidden_dim, n_heads):
        super(MultiAttention, self).__init__()

        self.build_input = InputBuilder(input_scheme, hidden_dim)
        n_var = self.build_input.n_var

        self.attn = MultiLayerAttention(n_layers, n_var, hidden_dim, n_heads)

    def forward(self, inputs):
        fixed_state, var_states = self.build_input(inputs)
        list_var_states = [var_states[k] for k in sorted(var_states.keys())]
        return self.attn(fixed_state, list_var_states)


class ActionAttention(nn.Module):
    def __init__(self, model_dim, n_actions):
        super(ActionAttention, self).__init__()

        self.model_dim = model_dim
        self.n_actions = n_actions
        self.fcq = nn.Linear(model_dim, model_dim * n_actions)
        self.fck = nn.Linear(model_dim, model_dim * n_actions)

    def forward(self, queries, keys):
        model_dim = self.model_dim
        n_actions = self.n_actions
        batch_size = queries.size(0)
        q = self.fcq(queries).view(batch_size * n_actions, 1, model_dim)
        k = self.fck(keys).view(batch_size, -1, n_actions, model_dim).transpose(1, 2).\
            reshape(batch_size * n_actions, -1, model_dim)
        v = th.bmm(q, k.transpose(1, 2)) / np.sqrt(model_dim)  # (batch * n_ac) * 1 * ?
        v = v.view(batch_size, n_actions, -1).transpose(1, 2)
        return v  # batch * ? * n_ac


class ActionAttentionV2(nn.Module):
    def __init__(self, model_dim, n_actions):
        super(ActionAttentionV2, self).__init__()

        self.model_dim = model_dim
        self.n_actions = n_actions
        self.fcq = nn.Linear(model_dim, model_dim)
        self.fck = nn.Linear(model_dim, model_dim)
        self.fca = nn.Linear(model_dim, n_actions)

    def forward(self, queries, keys):
        model_dim = self.model_dim
        n_actions = self.n_actions
        batch_size = queries.size(0)
        a = self.fca(queries)  # batch * n_ac
        q = self.fcq(queries).view(batch_size, 1, model_dim)
        k = self.fck(keys).view(batch_size, -1, model_dim)
        v = th.bmm(q, k.transpose(1, 2)) / np.sqrt(model_dim)  # batch * 1 * ?
        v = F.softmax(v, dim=-1)
        v = th.bmm(v.transpose(1, 2), a.view(batch_size, 1, n_actions))
        return v


class ActionAttentionV3(nn.Module):
    def __init__(self, model_dim, n_actions):
        super(ActionAttentionV3, self).__init__()

        self.model_dim = model_dim
        self.n_actions = n_actions
        self.fcq = nn.Linear(model_dim, model_dim)
        self.fck = nn.Linear(model_dim, model_dim)
        self.fca = nn.Linear(model_dim, n_actions)

    def forward(self, queries, keys):
        model_dim = self.model_dim
        n_actions = self.n_actions
        batch_size = queries.size(0)
        a = self.fca(queries)  # batch * n_ac
        q = self.fcq(queries).view(batch_size, 1, model_dim)  # batch * 1 * model_dim
        k = self.fck(keys).view(batch_size, -1, model_dim)  # batch * ? * model_dim
        v = th.bmm(q, k.transpose(1, 2)) / np.sqrt(model_dim)  # batch * 1 * ?
        v = F.tanh(v)
        v = th.bmm(v.transpose(1, 2), a.view(batch_size, 1, n_actions))
        return v


class ActionBuilder(nn.Module):
    def __init__(self, scheme, model_dim, version="v1"):
        super(ActionBuilder, self).__init__()

        actions = dict()

        idx = 0
        for part in scheme["action_pattern"]:
            if type(part) == int:
                key = None
                l = part
            else:
                l, key = part
            if key not in actions:
                actions[key] = {
                    "len": 0,
                    "split": [],
                    "idx": []
                }

            actions[key]["len"] += l
            actions[key]["split"].append(l)
            actions[key]["idx"].append(idx)
            idx += 1

        self.n_action_parts = idx
        var_keys = actions.keys() - {None}

        if None in actions:
            self.fixed_fc = nn.Linear(model_dim, actions[None]["len"])
        else:
            self.fixed_fc = None

        if version == "v1":
            action_attention_module = ActionAttention
        elif version == "v2":
            action_attention_module = ActionAttentionV2
        elif version == "v3":
            action_attention_module = ActionAttentionV3
        else:
            raise Exception("Version {} not supported!".format(version))

        var_fcs = dict()
        for key in var_keys:
            var_fcs[key] = action_attention_module(model_dim, actions[key]["len"])

        self.var_fcs = nn.ModuleDict(var_fcs)
        self.actions = actions

    def forward(self, h, var_states):
        batch_size = h.size(0)
        digits_parts = [None] * self.n_action_parts

        # def paint(input_digits, pos):
        #     idx = 0
        #     for s, t in pos:
        #         digits[s: t] = input_digits[idx: idx + t - s]
        #         idx += t - s

        def assign(digits, info):
            splits = digits.split(info["split"], dim=-1)
            for i, idx in enumerate(info["idx"]):
                digits_parts[idx] = splits[i].reshape(batch_size, -1)

        if self.fixed_fc is not None:
            fixed_digits = self.fixed_fc(h)
            assign(fixed_digits, self.actions[None])

        for key, attn in self.var_fcs.items():
            var_digits = attn(h, var_states[key])
            assign(var_digits, self.actions[key])

        return th.cat(digits_parts, dim=-1)
