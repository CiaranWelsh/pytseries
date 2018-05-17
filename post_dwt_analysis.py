import pandas, os
import sqlite3
import site

site.addsitedir('..')

dire = r'/home/b3053674/Documents/timeseries/Microarray'

db_file = os.path.join(dire, 'microarray_dwt.db')

conn = sqlite3.connect(db_file)

sql = """SELECT * FROM dwt_data ORDER BY cost"""
df = pandas.read_sql(sql, conn)


print (df)
x = 'CTGF'
y = 'FN1'

# def dtw1pair(df, x, y):
#     print(df[df['x'] == x])
    # d = timeseries(df[df['x_gene' == x]], df[df['y_gene'] == y], labels={'x': x, 'y': y})
    # d.distance_cost_plot()
    # d.dwt_plot()
    # plt.show()


# dtw1pair(df, 'CTGF.2', 'CTPS1.3')

# plt.figure()
# seaborn.distplot(df.cost)
#
# plt.show()
# from sklearn.cluster import KMeans
# c = KMeans(10)
# c.fit(df['cost'].reshape([-1, 1]))
#
# print(c.transform(df['cost'].reshape([-1, 1])), 'o')
#





