import pandas as pd
import matplotlib.pyplot as plt
import random
import scanpy
adata = scanpy.read_h5ad('Spatial-MGCN/result/Human_Breast_Cancer/Spatial_MGCN.h5ad')

# adata = scanpy.read_h5ad('Spatial-MGCN/result/DLPFC/151507/Spatial_MGCN.h5ad')




# print(adata.var_names)
# print(adata.var_names[:100])
# gene_name = "KRT8"  # 要检查的基因名称
# if gene_name in adata.var_names:
#     print(f"The gene {gene_name} is present in the dataset.")
# else:
#     print(f"The gene {gene_name} is not found in the dataset.")

# print("111")
# print(adata.obs['idx'])
# scanpy.pp.normalize_total(adata,target_sum=1e4)
# scanpy.pp.log1p(adata)

# scanpy.pp.normalize_total(adata,target_sum=100, exclude_highly_expressed=True)



# scanpy.pp.log1p(adata)
scanpy.pp.scale(adata,max_value=1)


# plt.rcParams.update({'font.size': 12})
scanpy.tools.rank_genes_groups(adata, "idx", method="wilcoxon")
scanpy.pl.rank_genes_groups_heatmap(adata, groups="0", n_genes=10, groupby="idx")


# 执行基因组排名分析
# scanpy.tl.rank_genes_groups(adata, "idx", method="t-test")
# # 定义颜色映射，使用更加直观的色谱
# cmap = plt.get_cmap("viridis")
# 创建热图
# ax = scanpy.pl.rank_genes_groups_heatmap(adata, groups="3", n_genes=10, groupby="idx", show_gene_labels=True,
#                                      cmap=cmap, figsize=(8, 6), standard_scale='var')
# scanpy.pl.heatmap(adata, var_names=['STMN1', 'COL9A2', 'CYP4B1', 'TTC39A', 'TACSTD2'],groupby='idx', cmap='viridis')