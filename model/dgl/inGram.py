import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import dgl
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
        self.act = nn.LeakyReLU(negative_slope=0.2)
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
        num_rel = self.args.aug_num_rels
        batch_num_edges = g.batch_num_edges()
        idx_list = torch.tensor(range(0, len(emb_rel), num_rel)).to(self.args.device)
        cum_idx = [torch.tensor(rel_idx).repeat_interleave(num_edge) for rel_idx, num_edge in zip(idx_list, batch_num_edges)]
        cum_idx = torch.concat(cum_idx).to(self.args.device)

        head_idxs, tail_idxs = g.edges()
        rel_idxs = g.edata['type'].clone()
        rel_idxs += cum_idx

        ent_freq = torch.zeros((num_ent,)).to(self.args.device).index_add(dim=0, index=tail_idxs, source=torch.ones_like(
            tail_idxs, dtype=torch.float).to(self.args.device)).unsqueeze(dim=1)
        self_rel = torch.zeros((num_ent, self.dim_rel)).to(self.args.device).index_add(dim=0, index=tail_idxs,
                                                                         source=emb_rel[rel_idxs]) / ent_freq

        # add self-loops
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim=0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent).to(self.args.device)], dim=0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent).to(self.args.device)], dim=0)

        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], emb_rels], dim=-1)
        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) *
                        self.attn_vec).sum(dim=-1, keepdim=True)
        scatter_idx = tail_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).to(self.args.device).scatter_reduce(dim=0, index=scatter_idx,
                                                                                      src=attn_val_raw, reduce='amax',
                                                                                      include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])
        attn_sums = torch.zeros((num_ent, self.num_head, 1)).to(self.args.device).index_add(dim=0, index=tail_idxs, source=attn_val)
        beta = attn_val / (attn_sums[tail_idxs] + 1e-16)
        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim=-1)
        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)
        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).to(self.args.device).index_add(dim=0, index=tail_idxs,
                                                                                          source=aggr_val)
        return output.flatten(1, -1)


class InGramRelationLayer(nn.Module):
    def __init__(self, args, dim_in_rel, dim_out_rel, num_bin, bias=True, num_head=8):
        super(InGramRelationLayer, self).__init__()
        self.args = args

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2 * dim_in_rel, dim_out_rel, bias=bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_bin, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, g):
        num_nodes = g.num_nodes()
        emb_rel = g.ndata.pop('h')
        head_idxs, tail_idxs = g.edges()
        bins = g.edata['bin']

        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim=-1)
        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) *
                        self.attn_vec).sum(dim=-1, keepdim=True) + self.attn_bin[bins]
        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        attn_val_max = torch.zeros((num_nodes, self.num_head, 1)).to(self.args.device).scatter_reduce(dim=0,
                                                                                      index=scatter_idx,
                                                                                      src=attn_val_raw, reduce='amax',
                                                                                      include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])
        attn_sums = torch.zeros((num_nodes, self.num_head, 1)).to(self.args.device).index_add(dim=0, index=head_idxs, source=attn_val)
        beta = attn_val / (attn_sums[head_idxs] + 1e-16)

        output = torch.zeros((num_nodes, self.num_head, self.dim_hid_rel)).to(self.args.device).index_add(
            dim=0, index=head_idxs,
            source=beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1, -1)


class Encoder(nn.Module):
    def __init__(self, args, bias=True):
        super(Encoder, self).__init__()
        self.args = args
        self.bias = bias
        self.num_layer_ent = args.num_layer_ent
        self.num_layer_rel = args.num_layer_rel
        self.act = nn.ReLU()

        # project layers
        self.ent_proj1 = nn.Linear(args.inp_dim, args.hid_dim_ent, bias=bias)
        # self.rel_proj1 = nn.Linear(args.rel_emb_dim, args.hid_dim_rel, bias=bias)

        # relation layers
        self.rel_emb = nn.Embedding(args.aug_num_rels, args.hid_dim_ent)

        layers_rel = []
        res_proj_rel = []
        for _ in range(args.num_layer_rel):
            layers_rel.append(InGramRelationLayer(args, args.hid_dim_rel, args.hid_dim_rel, args.num_bin, bias=bias, num_head=args.num_head))
        for _ in range(args.num_layer_rel):
            res_proj_rel.append(nn.Linear(args.hid_dim_rel, args.hid_dim_rel, bias=bias))
        self.layers_rel = nn.ModuleList(layers_rel)
        self.res_proj_rel = nn.ModuleList(res_proj_rel)

        # entity layers
        layers_ent = []
        res_proj_ent = []
        for _ in range(args.num_layer_ent):
            layers_ent.append(InGramEntityLayer(args, args.hid_dim_ent, args.hid_dim_ent, args.hid_dim_rel, bias=bias, num_head=args.num_head))
        for _ in range(args.num_layer_ent):
            res_proj_ent.append(nn.Linear(args.hid_dim_ent, args.hid_dim_ent, bias=bias))
        self.layers_ent = nn.ModuleList(layers_ent)
        self.res_proj_ent = nn.ModuleList(res_proj_ent)

        # self.ent_proj2 = nn.Linear(args.hid_dim_ent, args.emb_dim, bias=bias)
        # self.rel_proj2 = nn.Linear(args.hid_dim_rel, args.rel_emb_dim, bias=bias)

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.ent_proj1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.ent_proj2.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.rel_proj1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.rel_proj2.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.rel_proj.weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_ent):
            nn.init.xavier_normal_(self.res_proj_ent[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_rel):
            nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.ent_proj1.bias)
            # nn.init.zeros_(self.ent_proj2.bias)
            # nn.init.zeros_(self.rel_proj1.bias)
            # nn.init.zeros_(self.rel_proj2.bias)
            # nn.init.zeros_(self.rel_proj.bias)
            for layer_idx in range(self.num_layer_ent):
                nn.init.zeros_(self.res_proj_ent[layer_idx].bias)
            for layer_idx in range(self.num_layer_rel):
                nn.init.zeros_(self.res_proj_rel[layer_idx].bias)

    def forward(self, g, rel_g):
        # entity embedding
        feat = g.ndata.pop('feat')
        emb_ent = self.ent_proj1(feat)
        g.ndata['h'] = emb_ent
        # relation embedding
        feat_rel = rel_g.ndata.pop('feat')
        emb_rel = self.rel_emb(feat_rel) #self.rel_proj1(self.rel_emb(feat_rel))
        rel_g.ndata['h'] = emb_rel

        # relation layers
        for layer_idx, layer in enumerate(self.layers_rel):
            emb_rel = layer(rel_g) + self.res_proj_rel[layer_idx](emb_rel)
            emb_rel = self.act(emb_rel)
            rel_g.ndata['h'] = emb_rel

        # entity layers
        for layer_idx, layer in enumerate(self.layers_ent):
            emb_ent = layer(g, emb_rel) + self.res_proj_ent[layer_idx](emb_ent)
            emb_ent = self.act(emb_ent)
            g.ndata['h'] = emb_ent

        # emb_ent = g.ndata.pop('h')
        # emb_rel = rel_g.ndata.pop('h')
        # emb_ent = self.ent_proj2(emb_ent)
        # emb_rel = self.rel_proj2(emb_rel)

        return g.ndata.pop('h'), rel_g.ndata.pop('h')



class Model_Base(nn.Module):
    def __init__(self, args):
        super(Model_Base, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.readout = MaxPooling()

        if self.args.add_ht_emb:
            self.predictor = nn.Linear(3 * args.emb_dim + args.rel_emb_dim, 1)
        else:
            self.predictor = nn.Linear(args.emb_dim + args.rel_emb_dim, 1)

        nn.init.xavier_normal_(self.predictor.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.predictor.bias)

    def forward(self, data):
        g, rel_g, rel_labels_ = data
        emb_ent, emb_rel = self.encoder(g, rel_g)
        batch_num_nodes = rel_g.batch_num_nodes()
        batch_num_nodes = torch.cumsum(batch_num_nodes, dim=0)[:-1]
        rel_labels = torch.concat([torch.tensor([0]).to(self.args.device), batch_num_nodes])
        rel_labels += rel_labels_

        g_out = self.readout(g, emb_ent)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = emb_ent[head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = emb_ent[tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.emb_dim), head_embs.view(-1, self.args.emb_dim),
                               tail_embs.view(-1, self.args.emb_dim), emb_rel[rel_labels].view(-1, self.args.rel_emb_dim)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.emb_dim), emb_rel[rel_labels]], dim=1)

        output = self.predictor(g_rep)
        return output






