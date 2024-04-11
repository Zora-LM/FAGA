import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model

from model.dgl.graph_classifier import (GraphClassifier, GraphClassifier_InGram, GAE_FD, GAE_FD_before, GVAE_FD_simple,
                                        GVAE_FD, GVAE_FD_before, GVAE_FD_simple_v2)
from model.dgl.inGram import Model_Base
from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter
from utils.graph_utils import collate_fn, collate_fn_aug
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


dgl_model = {'RGCN': GraphClassifier,
             'InGram': GraphClassifier_InGram,
             'GAE_FD': GAE_FD,
             'GAE_FD_before': GAE_FD_before,
             'GVAE_FD_simple': GVAE_FD_simple,
             'GVAE_FD_simple_v2': GVAE_FD_simple_v2,
             'GVAE_FD': GVAE_FD,
             'GVAE_FD_before': GVAE_FD_before
             }


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir,
                                  f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_'
                                  f'neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)
        # exit()

    train = SubgraphDataset(params, params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params, params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model[params.model_name], params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid, collate_fn=collate_fn)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator, collate_fn=collate_fn_aug)

    logging.info('Starting training with full batch...')

    trainer.train()


def args():
    parser = argparse.ArgumentParser(description='TransE model')
    parser.add_argument("--device", type=str, default='cuda:0', help="Which GPU to use?")
    parser.add_argument("--model_name", type=str, default='GVAE_FD_simple_v2', choices=['RGCN', 'InGram'])
    parser.add_argument("--main_dir", type=str, default='./FAGA/')
    parser.add_argument("--exp_dir", type=str, default='./experiments/')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default='',
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default='NL-25', help="Dataset string")
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true', help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=10, help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3, help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=100,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000, help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4, help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=3,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--scheduler", type=str, default='Polynomial', choices=['MultiStepLR', 'Polynomial'])


    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations. Undirected')
    parser.add_argument('--add_reverse_rels', type=int, default=1,
                        help='whether to append adj matrix list with symmetric relations. Directed')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32, help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32, help="Entity embedding size")
    # parser.add_argument("--num_gcn_layers", "-l", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5, help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True, help='whether to have attn in model or not')

    parser.add_argument('--has_rel_graph', default=1, type=int)
    parser.add_argument('-nle', '--num_layer_ent', default=3, type=int)
    parser.add_argument('-nlr', '--num_layer_rel', default=1, type=int)
    parser.add_argument('-hdr_e', '--hid_dim_ent', default=128, type=int)
    parser.add_argument('-hdr_r', '--hid_dim_rel', default=128, type=int)
    parser.add_argument('--num_bin', default=10, type=int)
    parser.add_argument('--num_head', default=8, type=int)
    parser.add_argument('--res_rel', default=1, type=int)
    parser.add_argument('--res_ent', default=1, type=int)

    # for vae
    parser.add_argument('--mode', default='TransE', type=str, choices=['TransE', 'DistMult', 'RotatE', 'DistMult2'])
    # for self-supervised learning
    parser.add_argument('--ssl', default=0, type=int, help='Calculate contrastive loss')
    parser.add_argument('--factor', default=2, type=int)
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--edge_p', default=0.2, type=float, help='edge dropout rate')
    # for transfer
    parser.add_argument('--interp', default=1, type=int, help='linear interpolation ')
    parser.add_argument('--alpha', default=0.5, type=float, help='interpolation coefficient')
    parser.add_argument('--phi', default=2, type=float, help='interpolation coefficient')
    parser.add_argument('--gamma', default=0, type=float)

    params = parser.parse_args()

    return params

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    params = args()
    params.step = 'train'

    initialize_experiment(params, __file__)
    print(params.exp_dir)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    main(params)
    print(params.exp_dir)




