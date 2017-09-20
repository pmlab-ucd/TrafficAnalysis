import random
import numpy as np
import matplotlib.pyplot as plt
import pandas

import pylab


categories = list('ABCD')

# map categories to y-values
cat_dict = dict(zip(categories, range(1, len(categories)+1)))

# map y-values to categories
val_dict = dict(zip(range(1, len(categories)+1), categories))

# setup the dataframe
dates = pandas.DatetimeIndex(freq='20T', start='2012-05-05 13:00', end='2012-05-05 18:59')
values = [random.choice(categories) for _ in range(len(dates))]
df = pandas.DataFrame(data=values, index=dates, columns=['category'])

# determing the y-values from categories
df['plotval'] = df['category'].apply(cat_dict.get)

# make the plot
fig, ax = plt.subplots()
df['plotval'].plot(ax=ax, style='ks')
ax.margins(0.2)

# format y-ticks look up the categories
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: val_dict.get(x)))

pylab.show()