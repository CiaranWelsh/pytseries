import os, glob
import pandas, numpy
import matplotlib.pyplot as plt
import seaborn
import site
site.addsitedir(r'/home/b3053674/Documents')
from timeseries.core import TimeSeries, TimeSeriesGroup
from timeseries.clust import TimeSeriesKMeans




dire = r'/home/b3053674/Documents/timeseries/Microarray2/GOclustering'

microarray_data_file = r'/home/b3053674/Documents/timeseries/Microarray2/MeanMicroarrayDEGS_gt_50.csv'

data = pandas.read_csv(microarray_data_file, index_col=[0])
data.columns = [int(i) for i in data.columns]
data = data.dropna(how='all')

tsg = TimeSeriesGroup(data)
tsg = tsg.interpolate(num=30)
tsg = tsg.norm()


pattern = os.path.join(dire, 'GO*.csv')

files = glob.glob(pattern)

df_dct = {}
for i in files:
    go_term = os.path.split(i)[1][:-4]
    df_dct[go_term] = pandas.read_csv(i, index_col=[0])

go_df = pandas.concat(df_dct)
go_df.columns = ['genes']
# go_df.index = go_df.index.droplevel(1)

tgf_neg_reg = 'GO:0030512'

f = list(tsg.features)
# print (f)
def get_dups(go_id, tsg):
    """
    take genes associated with go_id that are
    in the data set tsg and expand on them to include
    duplicate probes which have the suffix gene.1 (for example
    :param go_id:
    :param tsg:
    :return:
    """
    dups = []
    for i in go_df.loc[go_id]:
        for j in go_df.loc[go_id, i]:
            for k in list(tsg.features):
                if j in k:
                    dups.append(k)

    return dups

def get_tsg(go_id, tsg):
    dups = get_dups(go_id, tsg)
    return TimeSeriesGroup(data.loc[dups])

go_df = go_df.reset_index()
go_df = go_df.drop('level_1', axis=1)
go_df.columns = ['go_id', 'gene']
print(go_df.go_id.unique())

go_map = {
    'GO:0030512': 'negative regulation of TGFb signalling',
    'GO:0030949': 'positive regulation of vascular endothelial growth factor receptor signaling pathway',
    'GO:0046330': 'positive regulation of JNK cascade',
    'GO:0046427': 'positive regulation of JAK-STAT cascade',
    'GO:0051897': 'positive regulation of protein kinase B signaling',
    'GO:0070373': 'negative regulation of ERK1 and ERK2 cascade',
    'GO:0070374': 'positive regulation of ERK1 and ERK2 cascade',
    'GO:1900745': 'positive regulation of p38MAPK cascade',
    # 'GO:1901223': 'negative regulation of NFkB signalling',
    # 'GO:1901224': 'positive regulation of NFkB signalling',
}
tsg_dct = {}
for label, df in go_df.groupby(by='go_id'):
    df = df.dropna(how='any')
    genes = list(df['gene'].values)
    tsg_dct[label] = TimeSeriesGroup(data.loc[genes])

from itertools import cycle
from cycler import cycler

lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

markers = list('.,ov^<>12348spP*hH+XxDd|_')
c = list('bgrcmyk')
# plt.rc('axes',
#        prop_cycle=(cycler('color', c) * cycler('linestyle', ['-','--', ':', '-.'])))
#                    cycler('marker', markers))

for i in tsg_dct:
    label = go_map[i]
    fname = os.path.join(dire, "{}.png".format(label))
    if len(tsg_dct[i]) <= 7:
        plt.rc('axes', prop_cycle=cycler('color', c))
    else:
        plt.rc('axes', prop_cycle=cycler('color', c) * cycler('linestyle', ['-', '--', ':', '-.']) )

    fig = tsg_dct[i].plot(tsg_dct[i].features, ylabel='FoldChange')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
#
# import matplotlib.pyplot as plt
# from cycler import cycler
# import numpy
# import seaborn
# seaborn.set_style('white')
#
#
# x = range(10)
# ys = []
# for i in range(20):
#     ys.append(numpy.random.uniform(1, 10, size=10)*i)
#
#
# plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) +
#                            cycler('linestyle', ['-', '--', ':', '-.', '-', '--'])))
#
# plt.figure()
# for i in range(20):
#     plt.plot(x, ys[i], label=i)
# plt.legend()
# plt.show()














