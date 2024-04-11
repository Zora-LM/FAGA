import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dgl.nn import AvgPooling, MaxPooling, SumPooling


class InGramEntityLayer(nn.Module):
    def __init__(self, args, dim_in_ent, dim_out_ent, dim_rel, bias=True, num_head=8):
        super(InGramEntityLayer, self).__init__()
        self.args = args

        self.dim_out_ent = dim_out_ent
        self.dim_hid_ent = dim_out_ent // num_head
        assert dim_out_ent == self.dim_hid_ent * num_head
        self.num_head = num_head

        self.attn_proj = nn.Linear(2 * dim_in_ent + dim_rel, dim_out_ent, bias=bias)
        self.attn_vec = nn.Parameter(torch.zeros((1, num_head, self.dim_hid_ent)))
        self.aggr_proj = nn.Linear(dim_in_ent + dim_rel, dim_out_ent, bias=bias)

        self.dim_rel = dim_rel
        self.act = nn.LeakyReLU(negative_slope = 0.2)
        self.bias = bias
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, g, emb_rel):
        emb_ent = g.ndata.pop('h')
        num_ent = g.num_nodes()

        head_idxs, tail_idxs = g.edges()
        rel_idxs = g.edata['type']

        ent_freq = torch.zeros((num_ent,)).to(self.args.device).index_add(
            dim=0, index=tail_idxs, source=torch.ones_like(tail_idxs, dtype=torch.float).to(self.args.device)).unsqueeze(dim=1)
        self_rel = torch.zeros((num_ent, self.dim_rel)).to(self.args.device).index_add(dim=0, index=tail_idxs,
                                                                                       source=emb_rel[rel_idxs]) / (ent_freq + 1e-6)

        # add self-loops
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim=0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent).to(self.args.device)], dim=0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent).to(self.args.device)], dim=0)

        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], emb_rels], dim=-1)
        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) *
                        self.attn_vec).sum(dim=-1, keepdim=True)
        scatter_idx = tail_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).to(self.args.device).scatter_reduce(dim=0,
                                                                                                    index=scatter_idx,
                                                                                                    src=attn_val_raw,
                                                                                                    reduce='amax',
                                                                                                    include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])
        attn_sums = torch.zeros((num_ent, self.num_head, 1)).to(self.args.device).index_add(dim=0, index=tail_idxs,
                                                                                            source=attn_val)
        beta = attn_val / (attn_sums[tail_idxs] + 1e-6)
        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim=-1)
        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)
        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).to(self.args.device).index_add(dim=0,
                                                                                                        index=tail_idxs,
                                                                                                        source=aggr_val)
        return output.flatten(1, -1)


class Encoder_Entity(nn.Module):
    def __init__(self, args, bias=True):
        super(Encoder_Entity, self).__init__()
        self.args = args
        self.max_label_value = args.max_label_value
        self.bias = bias
        self.num_layer_ent = args.num_layer_ent
        self.num_layer_rel = args.num_layer_rel
        self.act = nn.ReLU()

        # project layers
        # self.ent_proj1 = nn.Linear(args.inp_dim, args.hid_dim_ent, bias=bias)
        # self.ent_proj2 = nn.Linear(args.hid_dim_ent, args.emb_dim, bias=bias)
        # self.rel_proj = nn.Linear(args.rel_emb_dim, args.hid_dim_rel, bias=bias)

        # entity layers
        self.layers_ent = nn.ModuleList()
        self.res_proj_ent = nn.ModuleList()
        for _ in range(args.num_layer_ent):
            self.layers_ent.append(InGramEntityLayer(args, args.hid_dim_ent, args.hid_dim_ent, args.hid_dim_rel,
                                                bias=bias, num_head=args.num_head))
        if self.args.res_ent:
            for _ in range(args.num_layer_ent):
                self.res_proj_ent.append(nn.Linear(args.hid_dim_ent, args.hid_dim_ent, bias=bias))
                # self.res_proj_ent.append(nn.Identity())

        self.readout = AvgPooling()
        # self.dropout = nn.Dropout(args.dropout)

        self.param_init()

    def param_init(self):
        # nn.init.xavier_normal_(self.ent_proj1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.ent_proj2.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.rel_proj.weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_ent):
            nn.init.xavier_normal_(self.res_proj_ent[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            # nn.init.zeros_(self.ent_proj1.bias)
            # nn.init.zeros_(self.ent_proj2.bias)
            # nn.init.zeros_(self.rel_proj.bias)
            for layer_idx in range(self.num_layer_ent):
                nn.init.zeros_(self.res_proj_ent[layer_idx].bias)


    def forward(self, g, emb_rel):
        # entity embedding
        # feat = g.ndata.pop('feat')
        # emb_ent = self.act(self.ent_proj1(feat))
        # g.ndata['h'] = emb_ent
        # emb_rel = self.act(self.rel_proj(emb_rel))
        emb_ent = g.ndata['h']
        for layer_idx, layer in enumerate(self.layers_ent):
            h = layer(g, emb_rel)
            h = self.act(h)

            if self.args.res_ent:
                emb_ent = h + self.act(self.res_proj_ent[layer_idx](emb_ent))
            else:
                emb_ent = h

            # if layer_idx == len(self.layers_ent) - 1:
            #     emb_ent = self.dropout(emb_ent)
            g.ndata['h'] = emb_ent

        emb_ent = g.ndata.pop('h')
        # emb_ent = self.act(self.ent_proj2(emb_ent))
        g_out = self.readout(g, emb_ent)

        return emb_ent, g_out

