from pytseries import core
import os, glob, pandas
import seaborn
import matplotlib.pyplot as plt
import numpy

dire = r'/home/b3053674/Documents/pytseries/Microarray2'

results_dir = os.path.join(dire, 'PathwayEnrichment')
filenme = os.path.join(dire, 'MeanMicroarrayDEGS_gt_75.csv')

data = pandas.read_csv(filenme, index_col=0)

all_genes = list(data.index)

def get_dups(genes):
    l = []
    for g in genes:
        for i in all_genes:
            if g in i:
                l.append(i)

    return list(set(l))

from cycler import cycler
cycler_op2 = cycler('linestyle', ['-', '--', ':', '-.', '-', '--']) \
             * cycler('color', ['r', 'g', 'b', 'y', 'c', 'k'])
plt.rc('axes', prop_cycle=cycler_op2)

def pl(fname, genes):
    # genes = get_dups(genes)
    df = data.loc[genes].sort_index()
    print(df)
    tsg = core.TimeSeriesGroup(df)
    fig = tsg.plot(tsg.features, legend_loc=(1, -0.1))
    # plt.show()
    fig.savefig(fname, dpi=300, bbox_inches='tight')


fname = os.path.join(results_dir, 'reactome_raf_indep_reg_of_mapk1.png')
genes = ['DUSP10', 'DUSP1', 'DUSP2', 'DUSP5', 'DUSP6']
pl(fname, genes)

fname = os.path.join(results_dir, 'reactome_smad3_4.png')
genes = ['CDKN2', 'JUNB', 'SMAD7', 'MYC', 'SERPINE1', 'TGIF1']
pl(fname, genes)


fname = os.path.join(results_dir, 'reactome_down_reg_of_smad.png')
genes = ['SMURF2', 'SKI', 'SKIL', 'TGIF1', 'NCOR2']
pl(fname, genes)
##tgf b
fname = os.path.join(results_dir, 'kegg_tgf.png')
genes = ['BMP4', 'SMAD7', 'GDF6', 'INHBA', 'CDKN2B', 'ID2', 'ID1', 'TGIF1', 'ID4',
         'SMURF2', 'SMURF1', 'ID3', 'MYC']
pl(fname, genes)

##pi3k
fname = os.path.join(results_dir, 'kegg_pi3k.png')
genes = ['IL6','MCL1', 'PDGFA', 'ITGA2', 'NR4A1', 'TLR4', 'KIT',
         'IRS1', 'DDIT4', 'CDKN1B', 'LPAR6', 'COMP', 'COL27A1',
         'VEGFA', 'MYC','FN1', 'NGF']
pl(fname, genes)

##foxo
fname = os.path.join(results_dir, 'kegg_foxo.png')
genes = ['IL6', 'CDKN1B', 'PLK3', 'CDKN2B', 'FOXO1', 'GADD45B', 'KLF2','IRS1', 'CCNG2']
pl(fname, genes)

##tnf
fname = os.path.join(results_dir, 'kegg_tnf.png')
genes = ['LIF', 'CXCL1', 'IL6', 'CCL2','PTGS2', 'JUN', 'EDN1', 'JUNB']
pl(fname, genes)

##mapk
fname = os.path.join(results_dir, 'kegg_mapk.png')
genes = ['DUSP5', 'BDNF', 'DUSP2', 'DUSP1', 'PDGFA', 'JUN', 'DUSP10', 'NR4A1',
         'GADD45B', 'MYC', 'NGF','DUSP6']
pl(fname, genes)
##hif
fname = os.path.join(results_dir, 'kegg_hif.png')
genes = ['IL6', 'CDKN1B', 'PFKFB3','EDN1', 'VEGFA', 'SERPINE1', 'TLR4']

pl(fname, genes)







