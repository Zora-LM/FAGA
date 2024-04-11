import os
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl
import pickle
from utils.relgraph import generate_relation_graphs
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def process_files(params, files):

    data_name = params.dataset.replace('_ind', '')
    data = pickle.load(open(f'{params.main_dir}/data/{data_name}.pkl', 'rb'))
    entity2id = data['train_graph']['entity2id'] if 'ind' not in params.dataset else data['ind_test_graph']['entity2id']
    relation2id = data['relation2id']

    triplets = {}
    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for h, r, t in file_data:
            assert r in relation2id, print(f'Relation {r} does not exist!')
            data.append([entity2id[h], entity2id[t], relation2id[r]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation.
    # Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                        (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))),
                                       shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if params.add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    if params.add_reverse_rels:
        dgl_adj_list = ssp_multigraph_to_dgl_inv_edge(adj_list_aug)
        aug_num_rels = len(adj_list_aug)*2
    else:
        dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)
        aug_num_rels = len(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, aug_num_rels


# def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, aug_num_rels):
#     global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, aug_num_rels_
#     model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, aug_num_rels_ \
#         = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, aug_num_rels


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split('\n')[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet['head'][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet['tail'][1] = j

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None,
                                 node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def ssp_multigraph_to_dgl_inv_edge(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    num_rel = len(graph)
    # Add edges and reverse edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
            nx_triplets.append((dst, src, {'type': rel + num_rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, node_features=None, kge_entity2id=None):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params.hop,
                                                          enclosing_sub_graph=params.enclosing_sub_graph,
                                                          max_node_label_value=max_node_label_value)

        subgraph = dgl_adj_list.subgraph(nodes)
        # subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        try:
            edges_btw_roots = subgraph.edge_ids(0, 1)
        except:
            edges_btw_roots = []
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels)

def get_rank(params, neg_links, model, adj_list, dgl_adj_list, id2entity, node_features, kge_entity2id, rel_graph):
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs(head_neg_links, adj_list, dgl_adj_list, model.args.max_label_value, id2entity,
                             node_features, kge_entity2id)
        dgl_graphs, r_labels = data[0].to(params.device), data[1].to(params.device)
        head_scores = model((dgl_graphs, dgl_graphs.clone(), r_labels), rel_graph)[0].squeeze(1).detach().cpu().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs(tail_neg_links, adj_list, dgl_adj_list, model.args.max_label_value, id2entity,
                             node_features, kge_entity2id)
        dgl_graphs, r_labels = data[0].to(params.device), data[1].to(params.device)
        tail_scores = model((dgl_graphs, dgl_graphs.clone(), r_labels), rel_graph)[0].squeeze(1).detach().cpu().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join(params.experiment_name, f'{params.step}_ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join(params.experiment_name, f'{params.step}_ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join(params.experiment_name, f'{params.step}_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join(params.experiment_name, f'{params.step}_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def main(params):
    model = torch.load(params.model_path, map_location=params.device)
    model.args.device = params.device
    # model.args.res_ent = 1
    # model.args.res_rel = 1

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, aug_num_rels \
        = process_files(params, params.file_paths)
    rel_graph = generate_relation_graphs(dgl_adj_list, num_rel=aug_num_rels, B=params.num_bin)
    rel_graph = torch.tensor(rel_graph).to(params.device)

    node_features, kge_entity2id = get_kge_embeddings(params.dataset, params.kge_model) if params.use_kge_embeddings else (None, None)

    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
        save_to_file(neg_triplets, id2entity, id2relation)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    elif params.mode == 'ruleN':
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(params.ruleN_pred_path, entity2id, relation2id)

    ranks = []
    all_head_scores = []
    all_tail_scores = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(neg_triplets)):
                head_scores, head_rank, tail_scores, tail_rank = get_rank(params, batch, model, adj_list, dgl_adj_list,
                                                                          id2entity, node_features, kge_entity2id, rel_graph)
                ranks.append(head_rank)
                ranks.append(tail_rank)

                all_head_scores += head_scores.tolist()
                all_tail_scores += tail_scores.tolist()

    # num = 500
    # N = (len(neg_triplets) // num)
    # print(N)
    # n = 11
    # print('This is ', n)
    # with torch.no_grad():
    #     for i, batch in enumerate(tqdm(neg_triplets[num*n: num*(n+1)])):
    #         head_scores, head_rank, tail_scores, tail_rank = get_rank(params, batch, model, adj_list, dgl_adj_list,
    #                                                                   id2entity, node_features, kge_entity2id, rel_graph)
    #         ranks.append(head_rank)
    #         ranks.append(tail_rank)
    #
    #         all_head_scores += head_scores.tolist()
    #         all_tail_scores += tail_scores.tolist()
    #
    #     results = {'ranks': ranks, 'all_head_scores': all_head_scores, 'all_tail_scores': all_tail_scores}
    #     pickle.dump(results, open(f'{params.experiment_name}/{params.dataset}_{n}.pkl', 'wb'))


    # # with mp.Pool(processes=1, initializer=intialize_worker,
    # #              initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, aug_num_rels)) as p:
    # #     for head_scores, head_rank, tail_scores, tail_rank in tqdm(p.imap(get_rank, neg_triplets), total=len(neg_triplets)):
    # #         ranks.append(head_rank)
    # #         ranks.append(tail_rank)
    # #
    # #         all_head_scores += head_scores.tolist()
    # #         all_tail_scores += tail_scores.tolist()
    # #

    # for n in range(N):
    #     results = pickle.load(open(f'{params.experiment_name}/{params.dataset}_{n}.pkl', 'rb'))
    #     ranks += results['ranks']
    #     all_head_scores += results['all_head_scores']
    #     all_tail_scores += results['all_tail_scores']

    if params.mode == 'ruleN':
        save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)
    else:
        save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)

    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    logger.info('MRR | Hits@1 | Hits@5 | Hits@10 : %.4f,%.4f,%.4f,%.4f' % (mrr, hits_1, hits_5, hits_10))

    ls = [params.dataset, params.model_name, model.args.hop, model.args.res_ent, model.args.res_rel,
          model.args.num_layer_ent, model.args.num_layer_rel, model.args.num_bin,
          model.args.scheduler, model.args.lr, model.args.dropout, model.args.has_rel_graph, model.args.rel_emb_dim,
          model.args.hid_dim_ent, model.args.hid_dim_rel, model.args.ssl, model.args.tau, model.args.beta,
          model.args.edge_p, model.args.interp, model.args.alpha, model.args.phi, model.args.gamma, model.args.margin, model.args.l2,
          model.args.mode, model.args.num_neg_samples_per_link,
          round(mrr, 4), round(hits_1, 4), round(hits_5, 4), round(hits_10, 4)]
    f_csv = open(f'./experiments/{params.step}_resutls_mrr.csv', 'a')
    f_csv.write(','.join(map(str, ls)) + '\n')
    f_csv.close()

    # for n in range(N):
    #    os.remove(f'{params.experiment_name}/{params.dataset}_{n}.pkl')


def args():
    parser = argparse.ArgumentParser(description='Testing script for hits@10')
    parser.add_argument("--device", type=str, default='cuda:0', help="Which GPU to use?")
    parser.add_argument("--main_dir", type=str, default='/media/data2/lm/Experiments/KG/FD_dropout/')
    parser.add_argument("--model_name", type=str, default='GAE_FD', choices=['RGCN', 'InGram'])

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default='',
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="NL-100_ind", help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument('--add_reverse_rels', type=int, default=1,
                        help='whether to append adj matrix list with symmetric relations. Directed')
    parser.add_argument('--num_bin', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    params = parser.parse_args()

    return params


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    params = args()
    params.step = 'train'
    params.dataset = 'NL-50_ind'
    dataset_name = params.dataset.replace('_ind', '')
    params.experiment_name = (f'./experiments/{dataset_name}/{params.model_name}/hop2_nle3_nlr1_num_bin10_Polynomial_'
                              f'lr0.0001_l2_0.0001_dropout0.5_emb_rel32_hid_ent128_hid_rel128_has_rel_graph1_TransE_'
                              f'ssl0_factor2_tau0.1_beta1_edge_p0.2_'
                              f'interp0_alpha0.5_phi2_gamma0_margin1_neg1_res_ent1_res_rel1')

    ls = ['dataset', 'mode_name', 'hop', 'res_ent', 'res_rel', 'num_layer_ent', 'num_layer_rel', 'num_bin',
          'scheduler', 'lr', 'dropout', 'has_rel_graph', 'rel_emb_dim', 'hid_dim_ent',
          'hid_dim_rel', 'ssl', 'tau', 'beta', 'edge_p', 'interp', 'alpha', 'phi', 'gamma', 'margin', 'l2', 'mode', 'neg',
          'MRR', 'Hits@1', 'Hits@5', 'Hits@10']

    if not os.path.exists(f'./experiments/{params.step}_resutls_mrr.csv'):
        f_csv = open(f'./experiments/{params.step}_resutls_mrr.csv', 'a')
        f_csv.write(','.join(map(str, ls)) + '\n')
        f_csv.close()

    params.file_paths = {
        'graph': f'{params.main_dir}/data/{params.dataset}/train.txt',
        'links': f'{params.main_dir}/data/{params.dataset}/test.txt'
    }

    params.ruleN_pred_path = os.path.join('./data', params.dataset, 'pos_predictions.txt')
    params.model_path = os.path.join(params.experiment_name, f'{params.step}_best_graph_classifier.pth')

    file_handler = logging.FileHandler(os.path.join(params.experiment_name, f'log_rank_test.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)
    print(params.experiment_name)

