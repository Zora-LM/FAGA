import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InGramRelationLayer(nn.Module):
    def __init__(self, args, dim_in_rel, dim_out_rel, num_bin, bias=True, num_head=8):
        super(InGramRelationLayer, self).__init__()
        self.args = args

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2 * dim_in_rel, dim_out_rel, bias=bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        # self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, self.dim_hid_rel))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_bin, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)

        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim=-1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) *
                        self.attn_vec).sum(dim=-1, keepdim=True) + self.act(self.attn_bin[relation_triplets[..., 2]])

        # attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) *
        #                 self.attn_vec + self.attn_bin[relation_triplets[..., 2]]).sum(dim=-1, keepdim=True)

        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        # print(scatter_idx)
        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).to(self.args.device).scatter_reduce(dim=0,
                                                                                      index=scatter_idx,
                                                                                      src=attn_val_raw, reduce='amax',
                                                                                      include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1)).to(self.args.device).index_add(dim=0, index=head_idxs, source=attn_val)

        beta = attn_val / (attn_sums[head_idxs] + 1e-16)

        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).to(self.args.device).index_add(
            dim=0, index=head_idxs,
            source=beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1, -1)

class Encoder_Rel(nn.Module):
    def __init__(self, args, bias=True):
        super(Encoder_Rel, self).__init__()
        self.args = args
        self.bias = bias
        self.max_label_value = args.max_label_value
        self.num_bin = args.num_bin
        self.num_layer_rel = args.num_layer_rel
        self.act = nn.ReLU()

        self.rel_emb = nn.Embedding(args.aug_num_rels, args.rel_emb_dim)
        self.rel_proj1 = nn.Linear(args.rel_emb_dim, args.hid_dim_rel, bias=bias)

        self.layers_rel = nn.ModuleList()
        self.res_proj_rel = nn.ModuleList()
        for _ in range(args.num_layer_rel):
            self.layers_rel.append(InGramRelationLayer(args, args.hid_dim_rel, args.hid_dim_rel, args.num_bin,
                                                       bias=bias, num_head=args.num_head))
        if self.args.res_rel:
            for _ in range(args.num_layer_rel):
                self.res_proj_rel.append(nn.Linear(args.hid_dim_rel, args.hid_dim_rel, bias=bias))
                # self.res_proj_rel.append(nn.Identity())

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.rel_emb.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj1.weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_rel):
            nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.rel_proj1.bias)
            for layer_idx in range(self.num_layer_rel):
                nn.init.zeros_(self.res_proj_rel[layer_idx].bias)

    def forward(self, relation_triplets):

        emb_rel = self.act(self.rel_proj1(self.rel_emb.weight))

        for layer_idx, layer in enumerate(self.layers_rel):
            h = layer(emb_rel, relation_triplets)
            h = self.act(h)
            if self.args.res_rel:
                emb_rel = h + self.act(self.res_proj_rel[layer_idx](emb_rel))
            else:
                emb_rel = h

        return emb_rel

class InGramRelationLayer_v2(nn.Module):
    def __init__(self, args, dim_in_rel, dim_out_rel, bias=True, num_head=8):
        super(InGramRelationLayer_v2, self).__init__()
        self.args = args

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        # self.head_w = nn.Linear(dim_in_rel, dim_out_rel, bias=False)
        # self.tail_w = nn.Linear(dim_in_rel, dim_out_rel, bias=False)
        # self.bin_w = nn.Linear(dim_in_rel, dim_out_rel, bias=False)

        # self.attn_proj = nn.Linear(3 * dim_in_rel, dim_out_rel, bias=bias)
        self.attn_proj = nn.Linear(3 * dim_out_rel, dim_out_rel, bias=bias)
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.bias = bias

        self.param_init()

    def param_init(self):
        # nn.init.xavier_normal_(self.head_w.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.tail_w.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.bin_w.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_rel, emb_bin, relation_triplets):
        num_rel = len(emb_rel)

        head_idxs = relation_triplets[:, 0]
        tail_idxs = relation_triplets[:, 1]
        bin_idxs = relation_triplets[:, 2]

        # emb_head = self.head_w(emb_rel[head_idxs])
        # emb_tail = self.head_w(emb_rel[tail_idxs])
        # emb = self.head_w(emb_bin[bin_idxs])
        # concat_mat = torch.cat([emb_head, emb, emb_tail], dim=-1)
        concat_mat = torch.cat([emb_rel[head_idxs], emb_bin[bin_idxs], emb_rel[tail_idxs]], dim=-1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) *
                        self.attn_vec).sum(dim=-1, keepdim=True)

        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)
        # print(scatter_idx)
        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).to(self.args.device).scatter_reduce(dim=0,
                                                                                      index=scatter_idx,
                                                                                      src=attn_val_raw, reduce='amax',
                                                                                      include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1)).to(self.args.device).index_add(dim=0, index=head_idxs, source=attn_val)

        beta = attn_val / (attn_sums[head_idxs] + 1e-16)

        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).to(self.args.device).index_add(
            dim=0, index=head_idxs,
            source=beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1, -1)

class Encoder_Rel_v2(nn.Module):
    def __init__(self, args, bias=True):
        super(Encoder_Rel_v2, self).__init__()
        self.args = args
        self.bias = bias
        self.max_label_value = args.max_label_value
        self.num_bin = args.num_bin
        self.num_layer_rel = args.num_layer_rel
        self.act = nn.ReLU()

        self.rel_emb = nn.Embedding(args.aug_num_rels, args.rel_emb_dim)
        self.rel_proj1 = nn.Linear(args.rel_emb_dim, args.hid_dim_rel, bias=bias)

        self.bin_emb = nn.Embedding(args.num_bin, args.rel_emb_dim)
        self.bin_proj = nn.Linear(args.rel_emb_dim, args.hid_dim_rel, bias=bias)

        self.layers_rel = nn.ModuleList()
        self.res_proj_rel = nn.ModuleList()
        for _ in range(args.num_layer_rel):
            self.layers_rel.append(InGramRelationLayer_v2(args, args.hid_dim_rel, args.hid_dim_rel, bias=bias, num_head=args.num_head))
        if self.args.res_rel:
            for _ in range(args.num_layer_rel):
                self.res_proj_rel.append(nn.Linear(args.hid_dim_rel, args.hid_dim_rel, bias=bias))
                # self.res_proj_rel.append(nn.Identity())
        # self.dropout = nn.Dropout(args.dropout)

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.rel_emb.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.bin_emb.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.bin_proj.weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_rel):
            nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.rel_proj1.bias)
            nn.init.zeros_(self.bin_proj.bias)
            for layer_idx in range(self.num_layer_rel):
                nn.init.zeros_(self.res_proj_rel[layer_idx].bias)

    def forward(self, relation_triplets):

        emb_rel = self.act(self.rel_proj1(self.rel_emb.weight))
        emb_bin = self.act(self.bin_proj(self.bin_emb.weight))

        for layer_idx, layer in enumerate(self.layers_rel):
            h = layer(emb_rel, emb_bin, relation_triplets)
            h = self.act(h)

            if self.args.res_rel:
                emb_rel = h + self.act(self.res_proj_rel[layer_idx](emb_rel))
            else:
                emb_rel = h
            # if layer_idx == len(self.layers_rel)-1:
            #     emb_rel = self.dropout(emb_rel)

        return emb_rel


class Encoder_Rel_base(nn.Module):
    def __init__(self, args, bias=True):
        super().__init__()
        self.args = args
        self.bias = bias

        self.rel_emb = nn.Embedding(args.aug_num_rels, args.rel_emb_dim)
        self.rel_proj = nn.Linear(args.rel_emb_dim, args.hid_dim_rel, bias=bias)
        self.act = nn.ReLU()

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.rel_emb.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.rel_proj.bias)

    def forward(self, triplets):

        emb_rel = self.act(self.rel_proj(self.rel_emb.weight))

        return emb_rel


