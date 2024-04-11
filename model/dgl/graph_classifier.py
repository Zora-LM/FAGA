
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import dgl
from dgl import mean_nodes
from .rgcn_model import RGCN, RGCN2
from model.dgl.relation_graph_layer import Encoder_Rel, Encoder_Rel_v2, Encoder_Rel_base
from model.dgl.entity_encoding_layer import Encoder_Entity
from model.dgl.kge import MLP, TransE, DistMult, RotatE, DistMult2


score_method = {
    'MLP': MLP,
    'TransE': TransE,
    'DistMult': DistMult,
    'RotatE': RotatE,
    'DistMult2': DistMult2
}


class InfoNCE(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.args = args
        self.fc1 = torch.nn.Linear(dim, dim * args.factor)
        self.fc2 = torch.nn.Linear(dim * args.factor, dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.args.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.args.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)


class GraphClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_rel = Encoder_Rel(args=args)
        self.encoder_ent = RGCN(args)
        self.act = nn.ReLU()
        self.rel_proj = nn.Linear(args.hid_dim_rel, args.rel_emb_dim, bias=True)
        # self.rel_emb = nn.Embedding(self.args.num_rels, self.args.rel_emb_dim, sparse=False)

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.args.num_layer_ent * self.args.emb_dim + self.args.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.args.num_layer_ent * self.args.emb_dim + self.args.rel_emb_dim, 1)

    def forward(self, data, relation_triplets):
        emb_rel = self.encoder_rel(relation_triplets)
        emb_rel = self.act(self.rel_proj(emb_rel))

        g, rel_labels = data
        g.ndata['h'] = self.encoder_ent(g)
        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.num_layer_ent * self.args.emb_dim),
                               head_embs.view(-1, self.args.num_layer_ent * self.args.emb_dim),
                               tail_embs.view(-1, self.args.num_layer_ent * self.args.emb_dim),
                               emb_rel[rel_labels]], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.num_layer_ent * self.args.emb_dim),
                               emb_rel[rel_labels]], dim=1)

        output = self.fc_layer(g_rep)
        return output


class GraphClassifier_InGram(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        self.encoder_rel = Encoder_Rel(args=args)
        self.encoder_ent = Encoder_Entity(args)

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1)
        else:
            self.fc_layer = nn.Linear(args.hid_dim_ent + args.hid_dim_rel, 1)

    def forward(self, data, relation_triplets):
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output


class Decoder(nn.Module):
    def __init__(self, args, in_dim_ent, in_dim_rel, out_dim):
        super(Decoder, self).__init__()
        self.args = args
        self.head_fc = nn.Linear(in_dim_ent, out_dim)
        self.tail_fc = nn.Linear(in_dim_ent, out_dim)
        self.rel_fc = nn.Linear(in_dim_rel, out_dim)
        self.score = score_method[self.args.mode](out_dim)
        self.act = nn.ReLU()

    def forward(self, batch_g, emb_rel):
        z = batch_g.ndata.pop('z')
        head_idx, tail_idx = batch_g.edges()
        rel_idx = batch_g.edata['type']
        head_emb = self.act(self.head_fc(z[head_idx]))
        tail_emb = self.act(self.tail_fc(z[tail_idx]))
        rel_emb = self.act(self.rel_fc(emb_rel[rel_idx]))
        logit = self.score(head_emb, rel_emb, tail_emb)
        mask = torch.where(logit >= 0.5, 1, 0).squeeze()
        # try:
        batch_g.edata['mask'] = mask.view(-1)
        # except:
        #     print('Error')
        batch_g_list = dgl.unbatch(batch_g)

        # reconstruct graph
        recon_g_list = []
        for g in batch_g_list:
            num_nodes = g.num_nodes()
            label = g.edata.pop('label')
            edge_type = g.edata.pop('type')
            row, col = g.edges()
            edge_root = g.edge_ids(0, 1)
            mask = g.edata.pop('mask')
            mask[edge_root] = 1
            mask_idx = mask.nonzero().squeeze().view(-1)
            recon_g = dgl.graph(data=(row[mask_idx], col[mask_idx]), num_nodes=num_nodes, device=self.args.device)
            recon_g.ndata['id'] = g.ndata.pop('id')
            recon_g.edata['type'] = edge_type[mask_idx]
            recon_g.edata['label'] = label[mask_idx]
            recon_g_list.append(recon_g)

        return dgl.batch(recon_g_list)



class GAE_FD(nn.Module):
    def __init__(self, args):
        super(GAE_FD, self).__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        if args.has_rel_graph:
            self.encoder_rel = Encoder_Rel_v2(args=args)
        else:
            self.encoder_rel = Encoder_Rel_base(args=args)
        self.encoder_ent = Encoder_Entity(args)
        self.decoder = Decoder(args, in_dim_ent=args.hid_dim_ent, in_dim_rel=args.hid_dim_rel, out_dim=args.hid_dim_ent)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            # self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1),
            self.fc_layer = nn.Sequential(
                nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, args.hid_dim_ent),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hid_dim_ent, 1)
            )

        else:
            self.fc_layer = nn.Sequential(
                nn.Linear(args.hid_dim_ent + args.hid_dim_rel, args.hid_dim_ent),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hid_dim_ent, 1)
            )

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args, dim=args.hid_dim_ent)

        # self.enc_weights_mean = nn.Parameter(
        #     xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
        #     requires_grad=True)
        # self.enc_weights_std = nn.Parameter(
        #     xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
        #     requires_grad=True)

    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            # style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x
            style_applied = self.args.alpha * aug_x + (1 - self.args.alpha) * x

        # z_mean = torch.matmul(style_applied, self.enc_weights_mean)
        # z_std = torch.matmul(style_applied, self.enc_weights_std.float())
        # z_sampled = self.sampling(z_mean, z_std)

        # Reconstruct graph
        g.ndata['z'] = style_applied
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = style_applied
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        loss = self.Content_loss(x_recon, emb_ent) #(emb_ent + aug_emb_ent) * 0.5)
        style_loss = self.Style_loss(x_recon, emb_ent) #(emb_ent + aug_emb_ent) * 0.5)
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_eval_recon(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)

        # Reconstruct graph
        g.ndata['z'] = x
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        return output

    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def sampling(self, z_mean, z_std):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.args.hid_dim_ent).to(self.args.device)
        z_sampled = z_mean + torch.sqrt(torch.exp(z_std)) * eps
        return z_sampled

    def calc_mean_std(self, input, eps=1e-5):

        mean = torch.mean(input, dim=-1).view(-1, 1)  # Calculat mean
        std = torch.sqrt(torch.var(input, dim=-1) + eps).view(-1, 1)  # Calculate variance, add epsilon (avoid 0 division),
        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss



    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def calc_mean_std(self, input, eps=1e-5):
        '''
        Calculates mean and std dimension-wise
        '''
        mean = torch.mean(input, dim=-1).view(-1, 1)  # Calculat mean
        std = torch.sqrt(torch.var(input, dim=-1) + eps).view(-1, 1)  # Calculate variance, add epsilon (avoid 0 division),
        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss

class GAE_FD_before(nn.Module):
    def __init__(self, args):
        super(GAE_FD_before, self).__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        self.encoder_rel = Encoder_Rel(args=args)
        self.encoder_ent = Encoder_Entity(args)

        self.enc_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
                                     requires_grad=True)
        self.enc_weights_std = nn.Parameter(xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
                                    requires_grad=True)

        self.decoder = Decoder(args)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1)
        else:
            self.fc_layer = nn.Linear(args.hid_dim_ent + args.hid_dim_rel, 1)

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args)


    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x

        # Reconstruct graph
        g.ndata['z'] = style_applied
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent #style_applied
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        loss = self.Content_loss(x_recon, x)
        style_loss = self.Style_loss(x_recon, x)
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_eval_recon(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)

        # Reconstruct graph
        g.ndata['z'] = x
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        return output


    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def calc_mean_std(self, input):
        '''
        Calculates mean and std dimension-wise
        '''
        mean = torch.matmul(input, self.enc_weights_mean)
        std = torch.matmul(input, self.enc_weights_std.float())
        std = torch.sqrt(torch.exp(std))

        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss


class GVAE_RGCN(nn.Module):
    def __init__(self, args):
        super(GVAE_RGCN, self).__init__()
        self.args = args
        self.dim = args.emb_dim
        # self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        self.rel_proj = nn.Linear(args.hid_dim_rel, args.rel_emb_dim, bias=True)
        self.encoder_rel = Encoder_Rel(args=args)
        args.is_input_layer = True
        self.encoder_ent = RGCN2(args)
        self.decoder = Decoder(args, in_dim_ent=self.dim, in_dim_rel=self.dim, out_dim=self.dim)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.args.num_layer_ent * self.dim + args.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.args.num_layer_ent * self.dim + args.rel_emb_dim, 1)

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args, dim=self.dim)

        self.enc_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(self.dim, self.dim))), requires_grad=True)
        self.enc_weights_std = nn.Parameter(xavier_uniform_(torch.empty(size=(self.dim, self.dim))), requires_grad=True)

    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        emb_rel = self.act(self.rel_proj(emb_rel))

        # Encoder
        g, aug_g, rel_labels = data

        feat = g.ndata['feat'].clone()
        # feat = g.ndata.pop('feat')
        # emb_ent = self.act(self.ent_proj(feat))
        # g.ndata['h'] = emb_ent

        # aug_feat = aug_g.ndata.pop('feat')
        # aug_emb_ent = self.act(self.ent_proj(aug_feat))
        # aug_g.ndata['h'] = aug_emb_ent

        x = self.encoder_ent(g, emb_rel)
        g_out = mean_nodes(g, 'repr')
        aug_x = self.encoder_ent(aug_g, emb_rel)
        aug_g_out = mean_nodes(aug_g, 'repr')

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            # style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x
            style_applied = self.args.alpha * aug_x + (1 - self.args.alpha) * x

        z_mean = torch.matmul(style_applied, self.enc_weights_mean)
        z_std = torch.matmul(style_applied, self.enc_weights_std.float())
        z_sampled = self.sampling(z_mean, z_std)

        # Reconstruct graph
        g.ndata['z'] = z_sampled
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['feat'] = feat
        # recon_g.ndata['h'] = emb_ent  # style_applied
        # Encoder
        x_recon = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon
        recon_g_out = mean_nodes(recon_g, 'repr')

        # Predictor
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['repr'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['repr'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.num_layer_ent * self.dim),
                               head_embs.view(-1, self.args.num_layer_ent * self.dim),
                               tail_embs.view(-1, self.args.num_layer_ent * self.dim),
                               emb_rel[rel_labels]], dim=1)
        else:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.num_layer_ent * self.dim),
                               emb_rel[rel_labels]], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        # loss = F.mse_loss(x_recon, x)
        loss = torch.sum(torch.stack([F.mse_loss(recon_g.ndata['repr'][:, i, :],  g.ndata['repr'][:, i, :])
                                      for i in range(g.ndata['repr'].shape[1])]))

        # style_loss = self.Style_loss(x_recon, x)
        style_loss = torch.sum(torch.stack([self.Style_loss(recon_g.ndata['repr'][:, i, :],  g.ndata['repr'][:, i, :])
                                            for i in range(g.ndata['repr'].shape[1])]))
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        emb_rel = self.act(self.rel_proj(emb_rel))

        g, rel_labels = data
        # feat = g.ndata.pop('feat')
        # emb_ent = self.act(self.ent_proj(feat))
        # g.ndata['h'] = emb_ent
        g.ndata['h'] = self.encoder_ent(g, emb_rel)
        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.num_layer_ent * self.dim),
                               head_embs.view(-1, self.args.num_layer_ent * self.dim),
                               tail_embs.view(-1, self.args.num_layer_ent * self.dim),
                               emb_rel[rel_labels].view(-1, self.args.rel_emb_dim)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.dim), emb_rel[rel_labels].view(-1, self.args.rel_emb_dim)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def sampling(self, z_mean, z_std):
        eps = torch.randn(z_mean.size(0), self.dim).to(self.args.device)
        z_sampled = z_mean + torch.sqrt(torch.exp(z_std)) * eps
        return z_sampled

    def calc_mean_std(self, input, eps=1e-5):
        mean = torch.mean(input, dim=-1).view(-1, 1)  # Calculat mean
        std = torch.sqrt(torch.var(input, dim=-1) + eps).view(-1, 1)  # Calculate variance, add epsilon (avoid 0 division),
        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss

class GVAE_FD_simple_v2(nn.Module):
    def __init__(self, args):
        super(GVAE_FD_simple_v2, self).__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        if args.has_rel_graph:
            self.encoder_rel = Encoder_Rel_v2(args=args)
        else:
            self.encoder_rel = Encoder_Rel_base(args=args)
        self.encoder_ent = Encoder_Entity(args)
        self.decoder = Decoder(args, in_dim_ent=args.hid_dim_ent, in_dim_rel=args.hid_dim_rel, out_dim=args.hid_dim_ent)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            # self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1),
            self.fc_layer = nn.Sequential(
                nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, args.hid_dim_ent),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hid_dim_ent, 1)
            )

        else:
            self.fc_layer = nn.Sequential(
                nn.Linear(args.hid_dim_ent + args.hid_dim_rel, args.hid_dim_ent),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hid_dim_ent, 1)
            )

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args, dim=args.hid_dim_ent)

        self.enc_weights_mean = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.enc_weights_std = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)

    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            # style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x
            style_applied = self.args.alpha * aug_x + (1 - self.args.alpha) * x

        z_mean = torch.matmul(style_applied, self.enc_weights_mean)
        z_std = torch.matmul(style_applied, self.enc_weights_std.float())
        z_sampled = self.sampling(z_mean, z_std)

        # Reconstruct graph
        g.ndata['z'] = z_sampled
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = z_sampled
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        loss = self.Content_loss(x_recon, emb_ent) #(emb_ent + aug_emb_ent) * 0.5)
        style_loss = self.Style_loss(x_recon, emb_ent) #(emb_ent + aug_emb_ent) * 0.5)
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_recon(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            # style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x
            style_applied = self.args.alpha * aug_x + (1 - self.args.alpha) * x

        z_mean = torch.matmul(style_applied, self.enc_weights_mean)
        z_std = torch.matmul(style_applied, self.enc_weights_std.float())
        z_sampled = self.sampling(z_mean, z_std)

        # Reconstruct graph
        g.ndata['z'] = z_sampled
        recon_g = self.decoder(g, emb_rel)

        return (g, recon_g)


    def sampling(self, z_mean, z_std):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.args.hid_dim_ent).to(self.args.device)
        z_sampled = z_mean + torch.sqrt(torch.exp(z_std)) * eps
        return z_sampled

    def calc_mean_std(self, input, eps=1e-5):

        mean = torch.mean(input, dim=-1).view(-1, 1)  # Calculat mean
        std = torch.sqrt(torch.var(input, dim=-1) + eps).view(-1, 1)  # Calculate variance, add epsilon (avoid 0 division),
        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss


class GVAE_FD_simple(nn.Module):
    def __init__(self, args):
        super(GVAE_FD_simple, self).__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        self.encoder_rel = Encoder_Rel(args=args)
        self.encoder_ent = Encoder_Entity(args)
        self.decoder = Decoder(args, in_dim_ent=args.hid_dim_ent, in_dim_rel=args.hid_dim_rel, out_dim=args.hid_dim_ent)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1)
        else:
            self.fc_layer = nn.Linear(args.hid_dim_ent + args.hid_dim_rel, 1)

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args, dim=args.hid_dim_ent)

        self.enc_weights_mean = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.enc_weights_std = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)

    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            # style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x
            style_applied = self.args.alpha * aug_x + (1 - self.args.alpha) * x

        z_mean = torch.matmul(style_applied, self.enc_weights_mean)
        z_std = torch.matmul(style_applied, self.enc_weights_std.float())
        z_sampled = self.sampling(z_mean, z_std)

        # Reconstruct graph
        g.ndata['z'] = z_sampled
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent + aug_emb_ent
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        loss = self.Content_loss(x_recon, x + aug_emb_ent)
        style_loss = self.Style_loss(x_recon, x + aug_emb_ent)
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_eval_recon(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)

        # Reconstruct graph
        g.ndata['z'] = x
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        return output

    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def sampling(self, z_mean, z_std):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.args.hid_dim_ent).to(self.args.device)
        z_sampled = z_mean + torch.sqrt(torch.exp(z_std)) * eps
        return z_sampled

    def calc_mean_std(self, input, eps=1e-5):

        mean = torch.mean(input, dim=-1).view(-1, 1)  # Calculat mean
        std = torch.sqrt(torch.var(input, dim=-1) + eps).view(-1, 1)  # Calculate variance, add epsilon (avoid 0 division),
        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss


class GVAE_FD(nn.Module):
    def __init__(self, args):
        super(GVAE_FD, self).__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        self.encoder_rel = Encoder_Rel(args=args)
        self.encoder_ent = Encoder_Entity(args)
        self.decoder = Decoder(args)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1)
        else:
            self.fc_layer = nn.Linear(args.hid_dim_ent + args.hid_dim_rel, 1)

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args)

        self.enc_weights_mean = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.enc_weights_std = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
                                         requires_grad=True)
        self.weights_std = nn.Parameter(xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
                                        requires_grad=True)

    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        x_mean = torch.matmul(x, self.weights_mean)
        x_log_sigma_sq = torch.matmul(x, self.weights_std.float())
        z_sampled = self.sampling(x_mean, x_log_sigma_sq)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(z_sampled, x, aug_x)
        else:
            style_applied = self.args.alpha * self.AdaIn(z_sampled, x, aug_x) + (1 - self.args.alpha) * x

        # Reconstruct graph
        g.ndata['z'] = style_applied
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent #style_applied
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        loss = self.Content_loss(x_recon, x)
        style_loss = self.Style_loss(x_recon, x)
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_eval_recon(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)

        # Reconstruct graph
        g.ndata['z'] = x
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        return output

    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def sampling(self, z_mean, z_std):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.args.hid_dim_ent).to(self.args.device)
        z_sampled = z_mean + torch.sqrt(torch.exp(z_std)) * eps
        return z_sampled

    def calc_mean_std(self, input):
        '''
        Calculates mean and std dimension-wise
        '''
        mean = torch.matmul(input, self.enc_weights_mean)
        std = torch.matmul(input, self.enc_weights_std.float())
        std = torch.sqrt(torch.exp(std))

        return mean, std

    def AdaIn(self, z, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((z - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss


class GVAE_FD_before(nn.Module):
    def __init__(self, args):
        super(GVAE_FD_before, self).__init__()
        self.args = args
        self.ent_proj = nn.Linear(args.inp_dim, args.hid_dim_ent)
        self.encoder_rel = Encoder_Rel(args=args)
        self.encoder_ent = Encoder_Entity(args)
        self.decoder = Decoder(args)
        self.act = nn.ReLU()

        if self.args.add_ht_emb:
            self.fc_layer = nn.Linear(3 * args.hid_dim_ent + args.hid_dim_rel, 1)
        else:
            self.fc_layer = nn.Linear(args.hid_dim_ent + args.hid_dim_rel, 1)

        if self.args.ssl:
            self.contrastive = InfoNCE(args=args)

        self.enc_weights_mean = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.enc_weights_std = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.weights_mean = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)
        self.weights_std = nn.Parameter(
            xavier_uniform_(torch.empty(size=(self.args.hid_dim_ent, self.args.hid_dim_ent))),
            requires_grad=True)

    def forward(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, aug_g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        aug_feat = aug_g.ndata.pop('feat')
        aug_emb_ent = self.act(self.ent_proj(aug_feat))
        aug_g.ndata['h'] = aug_emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)
        aug_x, aug_g_out = self.encoder_ent(aug_g, emb_rel)

        # Transfer Style
        if not self.args.interp:
            style_applied = self.AdaIn(x, aug_x)
        else:
            style_applied = self.args.alpha * self.AdaIn(x, aug_x) + (1 - self.args.alpha) * x

        mean = torch.matmul(style_applied, self.weights_mean)
        std = torch.matmul(x, self.weights_std.float())
        z_sampled = self.sampling(mean, std)

        # Reconstruct graph
        g.ndata['z'] = z_sampled
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent  # style_applied
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)

        # Compute Loss
        loss = self.Content_loss(x_recon, x)
        style_loss = self.Style_loss(x_recon, x)
        loss += self.args.phi * style_loss

        return output, (g_out, aug_g_out), loss

    def forward_eval_recon(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)
        # Encoders
        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent

        x, g_out = self.encoder_ent(g, emb_rel)

        # Reconstruct graph
        g.ndata['z'] = x
        recon_g = self.decoder(g, emb_rel)
        recon_g.ndata['h'] = emb_ent
        # Encoder
        x_recon, recon_g_out = self.encoder_ent(recon_g, emb_rel)
        recon_g.ndata['h'] = x_recon

        # Predict
        head_ids = (recon_g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = recon_g.ndata['h'][head_ids]

        tail_ids = (recon_g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = recon_g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([recon_g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat(
                [recon_g_out.view(-1, self.args.hid_dim_ent), emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)],
                dim=1)

        output = self.fc_layer(g_rep)

        return output

    def forward_eval(self, data, relation_triplets):
        # relation embedding
        emb_rel = self.encoder_rel(relation_triplets)

        g, rel_labels = data
        feat = g.ndata.pop('feat')
        emb_ent = self.act(self.ent_proj(feat))
        g.ndata['h'] = emb_ent
        g.ndata['h'], g_out = self.encoder_ent(g, emb_rel)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['h'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['h'][tail_ids]

        if self.args.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent), head_embs.view(-1, self.args.hid_dim_ent),
                               tail_embs.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.args.hid_dim_ent),
                               emb_rel[rel_labels].view(-1, self.args.hid_dim_rel)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def sampling(self, z_mean, z_std):
        # pdb.set_trace()
        eps = torch.randn(z_mean.size(0), self.args.hid_dim_ent).to(self.args.device)
        z_sampled = z_mean + torch.sqrt(torch.exp(z_std)) * eps
        return z_sampled

    def calc_mean_std(self, input):
        '''
        Calculates mean and std dimension-wise
        '''
        mean = torch.matmul(input, self.enc_weights_mean)
        std = torch.matmul(input, self.enc_weights_std.float())
        std = torch.sqrt(torch.exp(std))

        return mean, std

    def AdaIn(self, content, style):
        assert content.shape == style.shape  # Only first two dim, such that different image sizes is possible
        mean_content, std_content = self.calc_mean_std(content)
        mean_style, std_style = self.calc_mean_std(style)

        output = std_style * ((content - mean_content) / (std_content)) + mean_style  # Normalise, then modify mean and std
        return output

    def Content_loss(self, input, target):  # Content loss is a simple MSE Loss
        loss = F.mse_loss(input, target)
        return loss

    def Style_loss(self, input, target):

        mean_input, std_input = self.calc_mean_std(input)
        mean_target, std_target = self.calc_mean_std(target)

        mean_loss = F.mse_loss(mean_input, mean_target)
        std_loss = F.mse_loss(std_input, std_target)

        return mean_loss + std_loss


