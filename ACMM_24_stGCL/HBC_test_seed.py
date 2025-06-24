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
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def load_data(dataset):
    print("load data:")
    path = "generate_data_3000/" + dataset + "/Spatial_MGCN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nsadj, nfadj, graph_nei, graph_neg


def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Human_Breast_Cancer']
    num_seeds = 10
    results = []

    for i in range(len(datasets)):
        dataset = datasets[i]
        path = 'Spatial-MGCN/result/' + dataset + '/'
        config_file = 'Spatial-MGCN/config/' + dataset + '.ini'
        if not os.path.exists(path):
            os.mkdir(path)
        print(dataset)
        adata, features, labels, sadj, fadj, graph_nei, graph_neg = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        savepath = 'Spatial-MGCN/result/Human_Breast_Cancer/'
        plt.rcParams["figure.figsize"] = (4, 3)

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        config.epochs = config.epochs + 1

        # rand_seeds = [300, 500, 700, 900, 1200, 1500, 1700, 2000, 2300, 2800, 3401, 5016, 6594, 8961, 151507]

        rand_seeds = [300, 500, 700, 900, 1200, 1500, 1700, 2000, 2300, 2800, 3401, 5016, 6594, 8961, 151507,100, 321, 523, 1423, 1773, 4364, 6143, 9056, 2539, 95463, 67354, 85346, 63428, 26456, 142316]
        # rand_seeds = [300]
        for seed in rand_seeds:
            # 设置随机种子
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True

            # 初始化模型和优化器
            model = Spatial_MGCN(nfeat=config.fdim,
                                 nhid1=config.nhid1,
                                 nhid2=config.nhid2,
                                 dropout=config.dropout)
            if cuda:
                model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

            ari_max = 0
            nmi_max = 0

            for epoch in range(config.epochs):
                emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train()
                kmeans = KMeans(n_clusters=config.class_num).fit(emb)
                idx = kmeans.labels_

                ari_res = metrics.adjusted_rand_score(labels, idx)
                nmi_res = metrics.normalized_mutual_info_score(labels, idx)
                print(f"Ari={ari_res}, NMI={nmi_res}")  # 同时在控制台打印结果
                # 更新最大ARI和NMI
                if ari_res > ari_max:
                    ari_max = ari_res
                    epoch_max = epoch
                    idx_max = idx
                    mean_max = mean
                    emb_max = emb
                if nmi_res > nmi_max:
                    nmi_max = nmi_res

            adata.obs['idx'] = idx_max.astype(str)
            adata.obsm['emb'] = emb_max
            adata.obsm['mean'] = mean_max

            # if config.gamma == 0:
            #     title = 'Spatial_MGCN-w/o'
            #     pd.DataFrame(emb_max).to_csv(savepath + 'Spatial_MGCN_no_emb.csv', header=None, index=None)
            #     pd.DataFrame(idx_max).to_csv(savepath + 'Spatial_MGCN_no_idx.csv', header=None, index=None)
            #     sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
            #     plt.savefig(savepath + 'Spatial_MGCN_no.jpg', bbox_inches='tight', dpi=600)
            #     # plt.show()
            # else:
            title = 'Spatial_MGCN'
            pd.DataFrame(emb_max).to_csv(savepath + 'Spatial_MGCN_emb.csv', header=None, index=None)
            pd.DataFrame(idx_max).to_csv(savepath + 'Spatial_MGCN_idx.csv', header=None, index=None)
            # sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)


            GT123 = labels
            id123 = adata.obs['idx']

            # 获取 ground_truth 和 pred_1 的类别名字
            ground_truth_classes = list(GT123.unique())  # ground_truth 的唯一类别名字
            pred_classes = list(id123.unique())  # pred_1 的唯一类别名字

            # 创建 ground_truth 和 pred_1 的映射字典
            ground_truth_to_int = {name: i for i, name in enumerate(ground_truth_classes)}
            int_to_ground_truth = {i: name for name, i in ground_truth_to_int.items()}
            pred_to_int = {name: i for i, name in enumerate(pred_classes)}
            int_to_pred = {i: name for name, i in pred_to_int.items()}
            # 将 ground_truth 和 pred_1 转换为整数
            true = [ground_truth_to_int[x] for x in GT123]
            pred_1 = [pred_to_int[x] for x in id123]
            # 计算混淆矩阵
            cm = confusion_matrix(true, pred_1)
            # 使用匈牙利算法进行标签重映射
            row_ind, col_ind = linear_sum_assignment(-cm)
            # 创建映射字典（pred_1 到 ground_truth 的整数标签重映射）
            label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
            # 根据映射重映射预测标签，并将其转回类别名字
            id123 = [int_to_ground_truth[label_mapping[label]] for label in pred_1]
            # 更新到 adata_Vars.obs
            adata.obs['idx'] = id123

            print(f"Ari={ari_max}, NMI={nmi_max}")  # 同时在控制台打印结果

            # sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=True)

            # sc.pl.spatial(adata, color=["stMMR", "Ground Truth"],
            #                 title=['stMMR (ARI=%.2f NMI=%.2f)' % (ARI, NMI), "Ground Truth"])

            # adata.layers['X'] = adata.X
            # adata.layers['mean'] = mean_max
            # plt.savefig(savepath + 'Spatial_MGCN.jpg', bbox_inches='tight', dpi=600)
            # plt.show()
            # adata.write(savepath + 'Spatial_MGCN.h5ad')
            
            # sc.pp.neighbors(adata, n_neighbors=15)
            # sc.tl.umap(adata)
            # sc.pl.umap(adata, color=['idx'], frameon = False,size=22)

            # 存储每个种子的最佳ARI和NMI
            results.append({'seed': seed, 'ari_max': ari_max, 'nmi_max': nmi_max})
            print(f"Seed {seed}: Best ARI = {ari_max}, Best NMI = {nmi_max}")

    # 打印所有结果
    for res in results:
        print(f"Seed {res['seed']}: Best ARI = {res['ari_max']}, Best NMI = {res['nmi_max']}")