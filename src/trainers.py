# import tqdm
import torch
import numpy as np
import logging
import os
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric
from utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.path = f'{args.output_dir}/{args.model_name}/{args.data_name}/{args.trial}'
        self.path = f'{args.output_dir}'
        if os.path.exists(self.path) is False:
            os.makedirs(self.path, exist_ok=True)
        self.writer = SummaryWriter(self.path)
        logging.basicConfig(filename=f'{self.path}/log.log', filemode='w', format = '%(message)s', level=logging.DEBUG)
        logging.info(f'model_args: {self.args}')
        print(f'model_args: {self.args}')
        logging.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

    def train(self):

        early_stopping = EarlyStopping(self.path+'/model.pt', patience=self.args.patience)
        full_sort = self.args.full_sort

        for epoch in range(self.args.epochs):
            self.iteration(epoch, self.train_dataloader)
            scores, result_info = self.valid(epoch, full_sort=full_sort)
            logging.info("Val, " + result_info)
            if full_sort:
                metric_dict = {
                        'val/HIT@5': scores[0], 'val/NDCG@5': scores[1], 
                        'val/HIT@10': scores[2], 'val/NDCG@10': scores[3], 
                        'val/HIT@20': scores[4], 'val/NDCG@20': scores[5], 
                        'val/HIT@30': scores[6], 'val/NDCG@30': scores[7],
                        'val/HIT@50': scores[8], 'val/NDCG@50': scores[9]
                        # 'val/HIT@100': scores[10], 'val/NDCG@100': scores[11]
                        }
            else:
                metric_dict = {'val/HIT@1': scores[0], 'val/NDCG@1': scores[1], 'val/HIT@5': scores[2], 'val/NDCG@5': scores[3], 'val/HIT@10': scores[4], 'val/NDCG@10': scores[5], 'val/MRR': scores[6]}
            [self.writer.add_scalar(key, metric_dict[key], global_step=epoch) for key in metric_dict]
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), self.model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(self.path+'/model.pt'))
        scores, result_info = self.test(0, full_sort=full_sort)
        logging.info("Test, " + result_info)
        args = vars(self.args)
        if full_sort:
            metric_dict = {
                    'test/HIT@5': scores[0], 'test/NDCG@5': scores[1], 
                    'test/HIT@10': scores[2], 'test/NDCG@10': scores[3], 
                    'test/HIT@20': scores[4], 'test/NDCG@20': scores[5], 
                    'test/HIT@30': scores[6], 'test/NDCG@30': scores[7],
                    'test/HIT@50': scores[8], 'test/NDCG@50': scores[9]
                    # 'test/HIT@100': scores[10], 'test/NDCG@100': scores[11]
                    }
            del args['train_matrix']
            del args['valid_rating_matrix']
            del args['test_rating_matrix']
        else:
            metric_dict = {'test/HIT@1': scores[0], 'test/NDCG@1': scores[1], 'test/HIT@5': scores[2], 'test/NDCG@5': scores[3], 'test/HIT@10': scores[4], 'test/NDCG@10': scores[5], 'test/MRR': scores[6]}
        [self.writer.add_scalar(key, metric_dict[key]) for key in metric_dict]

        self.writer.add_hparams(args, metric_dict)
        self.writer.close()

    def valid(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = "Epoch:{}, ".format(epoch) + \
            "HIT@1:{:.4f} ".format(HIT_1) + \
            "HIT@5:{:.4f} ".format(HIT_5) + \
            "HIT@10:{:.4f} ".format(HIT_10) + \
            "NDCG@1:{:.4f} ".format(NDCG_1) + \
            "NDCG@5:{:.4f} ".format(NDCG_5) + \
            "NDCG@10:{:.4f} ".format(NDCG_10) + \
            "MRR:{:.4f}".format(MRR)
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 20, 30, 50]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2]),
            "HIT@30": '{:.4f}'.format(recall[3]), "NDCG@30": '{:.4f}'.format(ndcg[3]),
            "HIT@50": '{:.4f}'.format(recall[4]), "NDCG@50": '{:.4f}'.format(ndcg[4])
            # "HIT@100": '{:.4f}'.format(recall[5]), "NDCG@100": '{:.4f}'.format(ndcg[5])
        }
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        print(original_state_dict.keys())
        new_dict = torch.load(file_name)
        print(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        #pos = pos_emb.view(-1, pos_emb.size(2))
        #neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out[:, -1, :] # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        #istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )# / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class PairwiseTrainer(Trainer):

    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(PairwiseTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in enumerate(dataloader):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, answer, neg_answer = batch
                # Binary cross_entropy
                sequence_output = self.model(input_ids)

                loss = self.cross_entropy(sequence_output, answer, neg_answer)

                self.optim.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     loss.backward()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = f"Train, Epoch:{epoch}, " + "rec_loss:{:.4f}".format(rec_loss)
            self.writer.add_scalar('train/loss', rec_loss)
            print(post_fix)

            logging.info(post_fix)

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in enumerate(dataloader):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, neg_answer = batch
                    recommend_output = self.model(input_ids)
                    recommend_output = recommend_output[:, -1, :]# 推荐的结果
                    
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -100)[:, -100:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in enumerate(dataloader):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, sample_negs = batch
                    recommend_output = self.model(input_ids)
                    test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
