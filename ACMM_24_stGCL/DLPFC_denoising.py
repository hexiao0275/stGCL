
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import os
import seaborn as sns

# adata = sc.read_h5ad('Spatial-MGCN/result/DLPFC/Spatial_MGCN.h5ad')
# adata = sc.read_h5ad('generate_data_3000/DLPFC/151507/Spatial_MGCN.h5ad')

# adata = sc.read_h5ad('Spatial-MGCN/result/DLPFC/151507/Spatial_MGCN.h5ad')
# adata = sc.read_h5ad('Spatial-MGCN/result/Human_Breast_Cancer/Spatial_MGCN.h5ad')
adata = sc.read_h5ad('Spatial-MGCN/result/Mouse_Brain_Anterior/Spatial_MGCN.h5ad')

savepath = 'Spatial-MGCN/result/DLPFC/denoise/'



if not os.path.exists(savepath):
    os.mkdir(savepath)

marker_genes = ['ATP2B4', 'FKBP1A', 'CRYM', 'NEFH', 'RXFP1', 'B3GALT2']#, 'NTNG2'

names=np.array(adata.var_names)

plt.rcParams["figure.figsize"] = (3, 3)
# for gene in marker_genes:
#     if np.isin(gene, np.array(adata.var_names)):
#         sc.pl.spatial(adata, img_key="hires", color=gene, show=False, title=gene, layer='X', vmax='p99')
#         plt.savefig(savepath + gene + '_raw.jpg', bbox_inches='tight', dpi=600)
#         plt.show()

#         sc.pl.spatial(adata, img_key="hires", color=gene, show=False, title=gene, layer='mean',
#                       vmax='p99')
#         plt.savefig(savepath + gene + '_mean.jpg', bbox_inches='tight', dpi=600)
#         plt.show()


# # Normalize the data using log1p and scaling
sc.pp.log1p(adata)
sc.pp.scale(adata)
# Rank genes using the Wilcoxon Rank-Sum Test
# idx ground

# sptial_2 = 'ground'
sptial_2 = 'idx'

sc.tl.rank_genes_groups(adata, sptial_2, method='wilcoxon')
top_genes = adata.uns['rank_genes_groups']['names']['3'][:10]
# viridis_r RdBu_r
# sc.pl.heatmap(adata, top_genes, groupby='ground', cmap='viridis', dendrogram=True)

sc.pl.rank_genes_groups_heatmap(
    adata,
    groupby='idx',
    n_genes=1,
    use_raw=False,
    swap_axes=True,
    show_gene_labels=True,
    vmin=-3,
    vmax=3,
    cmap="bwr",
)


plt.xlabel('Group')
plt.ylabel('Top Genes')
plt.show()

# ax = sc.pl.heatmap(
#     pbmc, marker_genes_dict, groupby="clusters", cmap="viridis", dendrogram=True
# )


# sc.pl.stacked_violin(adata1, marker_genes, title='Raw', groupby='ground_truth', swap_axes=True,
#                      figsize=[6, 3], show=False)
# plt.savefig(savepath + 'stacked_violin_Raw.jpg', bbox_inches='tight', dpi=10000)
# plt.show()

# sc.pl.stacked_violin(adata, marker_genes, layer='mean', title='Spatial-MGCN', groupby='ground_truth', swap_axes=True,
#                      figsize=[6, 3], show=False)
# plt.savefig(savepath + 'stacked_violin_Spatial_MGCN.jpg', bbox_inches='tight', dpi=3000)
# plt.show()