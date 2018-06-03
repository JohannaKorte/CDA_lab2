from preprocess import preprocess
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import pandas as pd


datafile = 'Data/BATADAL_train1.csv'
anomalydatafile = 'Data/BATADAL_train2.csv'
data = preprocess(datafile)
anomalydata = preprocess(anomalydatafile)
datadf = pd.DataFrame(data)

# PLOT AUTOCORRELATION
# ______________________________________________________________________________________________________________________

# Plot ACF and PACF to determine p and q
# lag_acf = acf(data[var], nlags=20)
# lag_pacf = pacf(data[var], nlags=20, method='ols')
#
# # Plot ACF
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
#
# # Plot PACF
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
# plt.show()

# TRAIN ARMA
# ______________________________________________________________________________________________________________________
# Order determined using the acf and pcf plots
order_dict = {'F_PU1': [9, 2]}


def train_arma(data, begin, end, order_dict, plot):
    """ Trains an ARMA model on the data between the 'begin' and 'end' slicing points using the given orders in
    order dict. """
    keys = list(order_dict.keys())
    for var in keys:
        model = ARMA(data[var][begin:end], order=(order_dict[var][0], order_dict[var][1]))
        results = model.fit(disp=-1)

        if plot:
            plt.plot(data['date'][begin:end], data[var][begin:end])
            plt.plot(data['date'][begin:end], results.fittedvalues, color='red')
            plt.show()
    return model

train_arma(data, -100, -1, order_dict, True)

# plt.plot(anomalydata['F_PU1'][:100])
# plt.plot(data['F_PU1'][:100])
# plt.show()

