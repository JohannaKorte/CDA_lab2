from preprocess import preprocess
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import *
import pandas as pd
import datetime


datafile = 'Data/BATADAL_train1.csv'
anomalydatafile = 'Data/BATADAL_train2.csv'
data = preprocess(datafile)
anomalydata = preprocess(anomalydatafile)
datadf = pd.DataFrame(data)

# PLOT AUTOCORRELATION
# ______________________________________________________________________________________________________________________


def plot_autocorr(data, var, lags):
    """ Plot ACF and PACF in order to determine p and q for the ARMA model."""
    lag_acf = acf(data[var], nlags=lags)
    lag_pacf = pacf(data[var], nlags=lags, method='ols')

    # Plot ACF
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data[var])),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

#plot_autocorr(data, 'F_PU11', 50)

# TRAIN ARMA
# ______________________________________________________________________________________________________________________
# Order determined using the acf and pcf plots
order_dict = {'F_PU1': [9, 2]}


def train_arma(data, var, begin, end, order_dict, plot):
    """ Trains an ARMA model on the data between the 'begin' and 'end' slicing points using the given orders in
    order dict. If plot is true, returns plots on fit and residuals"""
    model = ARMA(data[var], order=(order_dict[var][0], order_dict[var][1]))
    results = model.fit(disp=-1)

    if plot:
        # Plot fitted values together with data values
        plt.plot(data['date'], data[var], label='Data')
        plt.plot(data['date'], results.fittedvalues, color='red', label='ARMA')
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Signal')
        plt.title(var)
        plt.show()
        # Plot residuals
        plt.plot(data['date'], results.resid)
        plt.xlabel('Date')
        plt.ylabel('Residual error')
        plt.title(var)
        plt.show()
    return results

# train_arma(data, 'F_PU1', 0, len(data), order_dict, False)


def predict_arma(train, test, var, begin, end, order_dict):
    """ Given a train and test dataset, the sensor 'var' to be modelled, and the order dict, to determine the suitable
    ARMA order, returns a prediction for the test set from datapoint 50 onwards."""
    # Train ARMA model on train data and get its parameters
    results = train_arma(train, var, begin, end, order_dict, False)
    params = results.params
    # Setup a new ARMA model using the old parameters and the first 50 test variables.
    new_model = ARMA(test[var][:50], order=(order_dict[var][0], order_dict[var][1]))
    new_results = new_model.fit(start_params=params, disp=-1)
    new_prediction = new_results.forecast(steps=350)[0]
    plt.plot(range(400), test[var][:400])
    plt.plot(range(50,400), new_prediction)
    plt.show()
    return

print(predict_arma(data, anomalydata, 'F_PU1', 0, len(data), order_dict))
