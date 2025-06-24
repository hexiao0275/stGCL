
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import os

adata = sc.read_h5ad('Spatial-MGCN/result/DLPFC/Spatial_MGCN.h5ad')
adata1 = sc.read_h5ad('Spatial-MGCN/result/DLPFC/151507/Spatial_MGCN.h5ad')
savepath = 'Spatial-MGCN/result/DLPFC/denoise/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

marker_genes = ['ATP2B4', 'FKBP1A', 'CRYM', 'NEFH', 'RXFP1', 'B3GALT2']#, 'NTNG2'

names=np.array(adata.var_names)

# for gene in marker_genes:
#     if np.isin(gene, np.array(adata.var_names)):
#         sc.pl.spatial(adata, img_key="hires", color=gene, show=False, title=gene, layer='X', vmax='p99')
#         plt.savefig(savepath + gene + '_raw.jpg', bbox_inches='tight', dpi=600)
#         plt.show()

#         sc.pl.spatial(adata, img_key="hires", color=gene, show=False, title=gene, layer='mean',
#                       vmax='p99')
#         plt.savefig(savepath + gene + '_mean.jpg', bbox_inches='tight', dpi=600)
#         plt.show()

adata.obs = adata1.obs


sc.tl.rank_genes_groups(adata, groupby="idx", method="wilcoxon")

sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=50,
    values_to_plot="logfoldchanges",
    min_logfoldchange=3,
    vmax=7,
    vmin=-7,
    cmap="bwr",
)

