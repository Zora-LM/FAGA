import os
import logging
import json
import torch
import pickle


def initialize_experiment(params, file_name):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    # params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    dataset_name = params.dataset.replace('_ind', '')
    if 'RGCN' in params.model_name:
        params.exp_dir = (f'./experiments/{dataset_name}/{params.model_name}/hop{params.hop}_nle{params.num_layer_ent}_'
                          f'nlr{params.num_layer_rel}_num_bin{params.num_bin}_lr{params.lr}_emb_ent{params.emb_dim}_'
                          f'emb_rel{params.rel_emb_dim}_hid_rel{params.hid_dim_rel}'
                          f'_ssl{params.ssl}_factor{params.factor}_tau{params.tau}'
                          f'_beta{params.beta}_edge_p{params.edge_p}_interp{params.interp}_alpha{params.alpha}'
                          f'_phi{params.phi}_gamma{params.gamma}_margin{params.margin}'
                          f'_neg{params.num_neg_samples_per_link}_res_rel{params.res_rel}_multilayer')
    else:
        params.exp_dir = (
            f'./experiments/{dataset_name}/{params.model_name}/hop{params.hop}_nle{params.num_layer_ent}_'
            f'nlr{params.num_layer_rel}_num_bin{params.num_bin}_{params.scheduler}_lr{params.lr}_l2_{params.l2}_dropout{params.dropout}'
            f'_emb_rel{params.rel_emb_dim}_hid_ent{params.hid_dim_ent}_hid_rel{params.hid_dim_rel}_has_rel_graph{params.has_rel_graph}_'
            f'{params.mode}_ssl{params.ssl}_factor{params.factor}_tau{params.tau}_beta{params.beta}'
            f'_edge_p{params.edge_p}_interp{params.interp}_alpha{params.alpha}_phi{params.phi}_gamma{params.gamma}'
            f'_margin{params.margin}_neg{params.num_neg_samples_per_link}_res_ent{params.res_ent}_res_rel{params.res_rel}/')
    os.makedirs(params.exp_dir, exist_ok=True)

    if file_name == 'test_auc.py':
        params.test_exp_dir = os.path.join(params.exp_dir, f"test_{params.dataset}_{params.constrained_neg_prob}")
        if not os.path.exists(params.test_exp_dir):
            os.makedirs(params.test_exp_dir)
        file_handler = logging.FileHandler(os.path.join(params.test_exp_dir, f"log_test.txt"))
    else:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, f"log_{params.step}.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, f"{params.step}_params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, model, load_model=False):
    '''
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    '''

    if load_model and os.path.exists(os.path.join(params.exp_dir, params.model_pth)):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, params.model_pth))
        graph_classifier = torch.load(os.path.join(params.exp_dir, params.model_pth)).to(device=params.device)
    else:
        # data_name = params.dataset.replace('_ind', '')
        # data = pickle.load(open(f'./data/{data_name}.pkl', 'rb'))
        # relation2id = data['relation2id']
        # relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
        # with open(relation2id_path) as f:
        #     relation2id = json.load(f)

        logging.info('No existing model found. Initializing new model..')
        graph_classifier = model(params).to(device=params.device)

    return graph_classifier
