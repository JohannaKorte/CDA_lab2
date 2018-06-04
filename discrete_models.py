import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def discretize(data): # DatetimeParsed -> DiscData
    ''' Discretizes each of the columns into three bins. '''
    ndata = data.copy()
    for col in list(ndata):
        if col not in ['DATETIME', 'ATT_FLAG']:
            ndata[col] = pd.qcut(ndata[col], 3, labels=False, duplicates='drop')

    ndata = ndata.dropna(thresh=len(ndata) - 2, axis=1)
    return ndata

def dummies(data): # DiscData -> (DiscData, DummyData)
    ''' Creates dummies for the dataset elements. '''
    return data, pd.get_dummies(data, columns=list(data).remove('DATETIME'))

def read_datetime(data): # RawData -> DatetimeParsed
    ''' Properly parses the DATETIME field. '''
    ndata = data.copy()
    ndata['DATETIME'] = pd.to_datetime(ndata['DATETIME'], format='%d/%m/%y %H')
    return ndata

# data AnyData = RawData | DiscData | DummyData | DatetimeParsed
def drop_columns(data, cols=['DATETIME', 'ATT_FLAG']): # AnyData -> NoDatetimeData
    ''' Removes the defined columns. '''
    return data.drop(cols, axis=1)

def preprocess(data): # RawData -> (DiscData, DummyData)
    return dummies(discretize(read_datetime(data)))

# type AnomalyMarks = [Bool]
def pca_detect(data): # DummyData -> Sums
    ''' Uses PCA to find anomalies. '''
    pca = PCA()
    pca.fit(data)
    transformed = pca.transform(data) # map the dataset onto the PCA space
    return np.array([sum(x**2) for x in transformed/pca.singular_values_]) # calc Chi Squared value elements

data_train1 = pd.read_csv("./Data/BATADAL_train1.csv")
data_train2 = pd.read_csv("./Data/BATADAL_train2.csv", sep=', ')
data_train2['ATT_FLAG'] = data_train2['ATT_FLAG'] == 1
data_test = pd.read_csv("./Data/BATADAL_test.csv")

data_t1_disc, data_t1_dummies = preprocess(data_train1)
data_t2_disc, data_t2_dummies = preprocess(data_train2)

# anomaly detection
sums1 = pca_detect(drop_columns(data_t1_dummies))
hits1 = (sums1>0.25) == data_t1_dummies['ATT_FLAG']
sums2 = pca_detect(drop_columns(data_t2_dummies))
hits2 = (sums2>0.025) == data_t2_dummies['ATT_FLAG']

# plotting only 200 numbers
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
