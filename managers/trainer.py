import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics
from model.dgl.scheduler import PolynomialDecayLR
class Trainer():
    def __init__(self, args, graph_classifier, train, valid_evaluator=None, collate_fn=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.args = args
        self.train_data = train
        self.collate_fn = collate_fn(args)

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=self.args.l2)
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=args.lr, weight_decay=self.args.l2)
        step = len(train)//args.batch_size
        if args.scheduler == 'Polynomial':
            self.lr_scheduler = PolynomialDecayLR(self.optimizer, warmup_updates=step, tot_updates=step*10,
                                                  step_count=self.updates_counter, lr=args.lr, end_lr=1e-9, power=1)
        if args.scheduler == 'MultiStepLR':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[step, step*3, step*7], gamma=0.1)

        self.criterion = nn.MarginRankingLoss(self.args.margin, reduction='mean')

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers, collate_fn=self.collate_fn.collate_dgl)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = self.collate_fn.move_batch_to_device(batch)
            rel_graph = self.train_data.rel_graph.to(self.args.device)
            self.optimizer.zero_grad()
            score_pos, (g_out_pos, aug_g_out_pos), loss_recon_pos = self.graph_classifier(data_pos, rel_graph)
            score_neg, (g_out_neg, aug_g_out_neg), loss_recon_neg = self.graph_classifier(data_neg, rel_graph)
            # Losses
            loss = self.criterion(score_pos.view(len(score_pos)), score_neg.view(len(score_pos), -1).mean(dim=1),
                                  torch.Tensor([1]).to(device=self.args.device))
            loss += self.args.gamma * (loss_recon_pos + loss_recon_neg)
            if self.args.ssl:
                loss_ssl = self.graph_classifier.contrastive(g_out_pos, aug_g_out_pos)

                # loss_ssl = 0
                # for layer in range(g_out_pos.shape[1]):
                #     loss_ssl += self.graph_classifier.contrastive(g_out_pos[:, layer, :], aug_g_out_pos[:, layer, :])
                loss += self.args.beta * loss_ssl

            # print(score_pos, score_neg, loss)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.view(-1).detach().cpu().tolist() + score_neg.view(-1).detach().cpu().tolist()
                all_labels += targets_pos.view(-1).tolist() + targets_neg.view(-1).tolist()
                total_loss += loss

            if (self.valid_evaluator and self.args.eval_every_iter and
                    self.updates_counter % self.args.eval_every_iter == 0):
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.args.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.args.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        # print(all_scores)
        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.args.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # if self.valid_evaluator and epoch % self.args.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))
            
            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.args.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.args.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.args.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.args.exp_dir, f'{self.args.step}_graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.args.exp_dir, f'{self.args.step}_best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
