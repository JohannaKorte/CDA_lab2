import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

data_train1 = pd.read_csv("./Data/BATADAL_train1.csv")
data_train2 = pd.read_csv("./Data/BATADAL_train2.csv")
data_test = pd.read_csv("./Data/BATADAL_test.csv")
data_train1['DATETIME'] = pd.to_datetime(data_train1['DATETIME'], format='%d/%m/%y %H')

def discretize(data):
    ndata = data.copy()
    for col in list(ndata):
        if col not in ['DATETIME', 'ATT_FLAG']:
            ndata[col] = pd.qcut(ndata[col], 3, labels=False, duplicates='drop')

    ndata = ndata.dropna(thresh=len(ndata) - 2, axis=1)
    return ndata

def dummies(data):
    return pd.get_dummies(data, columns=list(data))

def preprocess(data):
    return dummies(discretize(data))

data_t1_disc = data_train1.drop(['DATETIME', 'ATT_FLAG'], axis=1) # datetime doesn't play nice with anything really
data_t1_disc = discretize(data_t1_disc)
data_t1_dummies = dummies(data_t1_disc)

pca = PCA()
pca.fit(data_t1_dummies)
transformed_t1 = pca.transform(data_t1_dummies) # map the dataset onto the PCA space
sums_t1 = np.array([sum(x**2) for x in transformed_t1/pca.singular_values_]) # calc Chi Squared value elements
anomalous = data_train1[sums_t1 > 0.25]

data_train1 = data_train1[0:200]
data_t1_disc = data_t1_disc[0:200]

# plotting the discretization
fig, ax1 = plt.subplots()
plt.plot(data_train1['DATETIME'], data_train1['F_PU1'])
ax1.set_xlabel('datetime')
ax2 = ax1.twinx()
plt.plot(data_train1['DATETIME'], data_t1_disc['F_PU1'], 'yo')
fig.tight_layout()
plt.savefig('discretization.png', pad_inches=0)
plt.clf()
