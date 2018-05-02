import pandas
import numpy
import os, glob
import seaborn
import deap


class FalseDF(object):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, item):
        return self.df.__getitem__(item)

    def __getattr__(self, item):
        return getattr(self.df, item)


df = pandas.DataFrame(numpy.reshape(numpy.arange(10), (2, 5)))
# print(df)
fdf = FalseDF(df)
# print(fdf[3])

print(fdf.loc[1])















