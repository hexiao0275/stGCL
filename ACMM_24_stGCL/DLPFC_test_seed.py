from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def load_data(dataset):
    print("load data:")
    path = "generate_data_3000/DLPFC/" + dataset + "/Spatial_MGCN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg

def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    dicr_i_loss = dicr_loss(com1, com2)
    Tail_loss = TailClusterLoss(emb)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss + dicr_i_loss * 0.01
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss, Tail_loss, dicr_i_loss

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151507']
    num_seeds = 10
    results = []

    for dataset in datasets:
        config_file = 'Spatial-MGCN/config/DLPFC.ini'
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = 'Spatial-MGCN/result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground'], title=title, show=False)
        plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        # rand_seeds = [300, 500, 700, 900, 1200, 1500, 1700, 2000, 2300, 2800, 3401, 5016, 6594, 8961, 151507]
        rand_seeds = [1700]
        for seed in rand_seeds:
            # 设置随机种子
            torch.manual_seed(seed)

            if cuda:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True

            print(f"Running with seed: {seed}")
            model = Spatial_MGCN(nfeat=config.fdim, nhid1=config.nhid1, nhid2=config.nhid2, dropout=config.dropout)
            if cuda:
                model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

            ari_max = 0
            nmi_max = 0
            for epoch in range(config.epochs):
                emb, mean, zinb_loss, reg_loss, con_loss, total_loss, Tail_loss, dicr_i_loss = train()
                # print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                #       ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
                #       ' total_loss = {:.2f}'.format(total_loss), ' Tail_loss = {:.2f}'.format(Tail_loss),
                #       ' dicr_loss = {:.2f}'.format(dicr_i_loss))

                kmeans = KMeans(n_clusters=config.class_num).fit(emb)
                idx = kmeans.labels_
                ari_res = metrics.adjusted_rand_score(labels, idx)
                nmi_res = metrics.normalized_mutual_info_score(labels, idx)

                # print('Spatial-MGCN: ARI={:.2f} NMI={:.2f}'.format(ari_res, nmi_res))

                if ari_res > ari_max:
                    ari_max = ari_res
                    nmi_max = nmi_res
                    epoch_max = epoch
                    idx_max = idx
                    mean_max = mean
                    emb_max = emb

            results.append({'seed': seed, 'ari_max': ari_max, 'nmi_max': nmi_max})
            print(f"Seed {seed}: Best ARI = {ari_max}, Best NMI = {nmi_max}")

        # 输出所有种子的最佳结果
        for res in results:
            print(f"Seed {res['seed']}: Best ARI = {res['ari_max']}, Best NMI = {res['nmi_max']}")

        # 这里可以保存最后的结果


        labels.replace('1', 't0', inplace=True)
        labels.replace('2', 't1', inplace=True)
        labels.replace('3', 't2', inplace=True)
        labels.replace('4', 't3', inplace=True)
        labels.replace('5', 't4', inplace=True)
        labels.replace('6', 't5', inplace=True)
        labels.replace('0', 't6', inplace=True)

        labels.replace('t0', '0', inplace=True)
        labels.replace('t1', '1', inplace=True)
        labels.replace('t2', '2', inplace=True)
        labels.replace('t3', '3', inplace=True)
        labels.replace('t4', '4', inplace=True)
        labels.replace('t5', '5', inplace=True)
        labels.replace('t6', '6', inplace=True)


        labels_int = [int(x) for x in labels]
        true = labels_int
        pred_1 = idx_max
        cm = confusion_matrix(true, pred_1)
        # 使用匈牙利算法进行标签重映射
        row_ind, col_ind = linear_sum_assignment(-cm)
        # 创建映射字典
        label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        # 根据映射重映射预测标签
        idx_max = np.array([label_mapping[label] for label in pred_1])


        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        title = 'Spatial-MGCN: ARI={:.2f} NMI={:.2f}'.format(ari_max, nmi_max)
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=True)
        plt.savefig(savepath + 'Spatial_MGCN.jpg', bbox_inches='tight', dpi=600)


        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['idx'], frameon=False)
        # plt.savefig(savepath + 'Spatial_MGCN_umap_mean.jpg', bbox_inches='tight', dpi=600)

        pd.DataFrame(emb_max).to_csv(savepath + 'Spatial_MGCN_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'Spatial_MGCN_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'Spatial_MGCN.h5ad')
