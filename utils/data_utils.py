import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import pickle


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    rel_reidx = dict()

    entidx = 0
    relidx = 0

    # rel_reidx['self'] = relidx
    # relidx += 1

    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])

    # for h in ent_reidx.keys():
    #     tri_reidx.append([ent_reidx[h], rel_reidx['self'], ent_reidx[h]])

    return tri_reidx, dict(rel_reidx), dict(ent_reidx)


def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0

    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])

    # for h in ent_reidx.keys():
    #     tri_reidx.append([ent_reidx[h], rel_reidx['self'], ent_reidx[h]])

    return tri_reidx, dict(ent_reidx)

def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx


def data2pkl(root_dir, data_name):
    # pdb.set_trace()
    train_tri = []
    file = open('{}/data/{}/train.txt'.format(root_dir, data_name))
    train_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    valid_tri = []
    file = open('{}/data/{}/valid.txt'.format(root_dir, data_name))
    valid_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    test_tri = []
    file = open('{}/data/{}/test.txt'.format(root_dir, data_name))
    test_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    file = open('{}/data/{}_ind/train.txt'.format(root_dir, data_name))
    ind_train_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('{}/data/{}_ind/valid.txt'.format(root_dir, data_name))
    ind_valid_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('{}/data/{}_ind/test.txt'.format(root_dir, data_name))
    ind_test_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
    test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri = reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri, 'entity2id': ent_reidx},
                 'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri, 'entity2id': ent_reidx_ind},
                 'relation2id': fix_rel_reidx}

    pickle.dump(save_data, open(f'{root_dir}/data/{data_name}.pkl', 'wb'))


def data2pkl_new_rel(root_dir, data_name):

    all_tri = []
    msg_tri = []
    file = open('{}/data/{}/msg.txt'.format(root_dir, data_name))
    msg_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()
    all_tri.extend(msg_tri)

    train_tri = []
    file = open('{}/data/{}/train.txt'.format(root_dir, data_name))
    train_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()
    all_tri.extend(train_tri)

    valid_tri = []
    file = open('{}/data/{}/valid.txt'.format(root_dir, data_name))
    valid_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()
    all_tri.extend(valid_tri)

    test_tri = []
    file = open('{}/data/{}/test.txt'.format(root_dir, data_name))
    test_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()
    all_tri.extend(test_tri)

    _, fix_rel_reidx, _ = reidx(all_tri)

    # train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    _, ent_reidx = reidx_withr(train_tri + valid_tri, fix_rel_reidx)
    train_tri = reidx_withr_ande(train_tri, fix_rel_reidx, ent_reidx)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    # test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    file = open('{}/data/{}/msg.txt'.format(root_dir, data_name))
    ind_train_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    # file = open('{}/data/{}_ind/valid.txt'.format(root_dir, data_name))
    # ind_valid_tri = ([l.strip().split() for l in file.readlines()])
    # file.close()

    file = open('{}/data/{}/test.txt'.format(root_dir, data_name))
    ind_test_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    _, ent_reidx_ind = reidx_withr(ind_train_tri + ind_test_tri, fix_rel_reidx)
    test_train_tri = reidx_withr_ande(ind_train_tri, fix_rel_reidx, ent_reidx_ind)
    # test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri = reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'entity2id': ent_reidx},
                 'ind_test_graph': {'train': test_train_tri, 'test': test_test_tri, 'entity2id': ent_reidx_ind},
                 'relation2id': fix_rel_reidx}

    pickle.dump(save_data, open(f'{root_dir}/data/{data_name}.pkl', 'wb'))

def process_files_(files, entity2id, relation2id):
    '''
    files: Dictionary map of file paths to read the triplets from.
    '''

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

    # Construct the list of adjacency matrix each corresponding to a relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, id2entity, id2relation


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations
    to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to a relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
