__author__ = 'robertdavis'
# Example Modeled after:
#    http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html

from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = load_data()

num_cols = ['Market Capitalization', 'Earnings/Share', 'Dividend Yield', 'EBITDA', 'EPS Estimate Current Year',
            'EPS Estimate Next Quarter', 'EPS Estimate Next Year', '200-day Moving Average', '50-day Moving Average',
            '52-week High', '52-week Low', '52-week Range', 'Price/Book', 'Book Value', 'Volume',
            'Average Daily Volume', 'Short Ratio', 'P/E Ratio', 'PEG Ratio', 'Ask', 'Open', 'Previous Close',
            'Change in Percent', 'Day Range']
numeric_cols = []

for col in num_cols:
    if type(data[col][0]) == np.float64:
        numeric_cols.append(col)
data_filter = data[numeric_cols].dropna()

y_pred = KMeans(n_clusters=10).fit_predict(data_filter)


