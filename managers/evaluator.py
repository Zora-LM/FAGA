import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator():
    def __init__(self, args, graph_classifier, data, collate_fn=None):
        self.args = args
        self.graph_classifier = graph_classifier
        self.data = data
        self.collate_fn = collate_fn(args)


    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.args.batch_size, shuffle=False, 
                                num_workers=self.args.num_workers, collate_fn=self.collate_fn.collate_dgl)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                data_pos, targets_pos, data_neg, targets_neg = self.collate_fn.move_batch_to_device(batch)
                rel_graph = self.data.rel_graph.to(self.args.device)
                data_pos = data_pos[0], data_pos[0].clone(), data_pos[1]
                data_neg = data_neg[0], data_neg[0].clone(), data_neg[1]
                score_pos, _, _ = self.graph_classifier(data_pos, rel_graph)
                score_neg, _, _ = self.graph_classifier(data_neg, rel_graph)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()

        # acc = metrics.accuracy_score(labels, preds)
        # print(pos_scores)
        # print(neg_scores)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.args.main_dir, 'data/{}/{}.txt'.format(self.args.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = f'{self.args.exp_dir}/{self.args.step}_{self.data.file_name}_predictions.txt'
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = f'{self.args.exp_dir}/neg_{self.args.step}_{self.data.file_name}_{self.args.constrained_neg_prob}.txt'
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.args.exp_dir, 'neg_{}_{}_{}_predictions.txt'.format(
                self.args.step, self.data.file_name, self.args.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}


    def extract_subgraph(self):

        dataloader = DataLoader(self.data, batch_size=self.args.batch_size, shuffle=False,
                                num_workers=self.args.num_workers, collate_fn=self.collate_fn.collate_dgl)

        self.graph_classifier.eval()

        subgraphs = []
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                data_pos, targets_pos, _, _ = self.collate_fn.move_batch_to_device(batch)
                rel_graph = self.data.rel_graph.to(self.args.device)
                data_pos = data_pos[0], data_pos[0].clone(), data_pos[1]
                g, recon_g = self.graph_classifier.forward_recon(data_pos, rel_graph)
                subgraphs.append([g.to('cpu'), recon_g.to('cpu')])

        return subgraphs

