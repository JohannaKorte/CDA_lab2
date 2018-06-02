import preprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

filename = 'Data/BATADAL_train1.csv'
data = preprocess.preprocess(filename)


def visualize(data):
    """ Visualizes the behavior of all signals over time in separate figures. """
    for key in data.keys():
        if key == 'data':
            continue
        else:
            plt.plot(data['date'][:1000], data[key][:1000])
            plt.xlabel('Date')
            plt.ylabel('Signal')
            plt.title(key)
            plt.show()


def correlate(data):
    """ Calculates the correlation between every pair of signals"""
    correlation_matrix = []
    ticks = list(data.keys())[1:-1]
    for key1 in data.keys():
        correlation = []
        for key2 in data.keys():
            if key1 != 'date' and key2 != 'date' and key1 != 'flag' and key2 != 'flag':
                correlation.append(pearsonr(data[key1], data[key2])[0])
        corrected_correlation = []
        for x in correlation:
            if np.isnan(x) == False:
                corrected_correlation.append(x)
            else:
                corrected_correlation.append(0.0)
        if corrected_correlation != []:
            correlation_matrix.append(corrected_correlation)
    correlation_matrix = np.array(correlation_matrix)

    # Plot correlation matrix
    fig, ax = plt.subplots()
    im = ax.imshow(correlation_matrix)
    ax.set_xticks(np.arange(len(data.keys())-2))
    ax.set_yticks(np.arange(len(data.keys())-2))
    ax.set_xticklabels(ticks, {'fontsize': 6})
    ax.set_yticklabels(ticks, {'fontsize': 6})
    plt.xlabel('Sensors')
    plt.ylabel('Sensors')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Pearson correlation', rotation=-90, va="bottom")
    plt.show()
    return


def plot_corrs(data, sensors):
    """Plots the first 1000 timepoints of the given sensors to see if there is a correlation"""
    x = data['date']
    # print(pearsonr(data[sensors[0]], data[sensors[1]])[0])
    # print(pearsonr(data[sensors[0]], data[sensors[2]])[0])
    # print(pearsonr(data[sensors[1]], data[sensors[2]])[0])
    fig, ax = plt.subplots()
    for sensor in sensors:
        y = data[sensor]
        ax.plot(x[:1000], y[:1000])
    plt.xlabel('Date')
    plt.ylabel('Signal strength')
    ax.legend(sensors, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
             rotation_mode="anchor", fontsize=5)
    plt.show()


def predict(data, signal, vis, w):
    datadf = data[signal]
    df = pd.DataFrame({'Count': datadf[:1100]})
    train = df[:1000]
    test = df[999:1100]
    mov_av = test.copy()
    mov_av['Count'] = train['Count'].rolling(w).mean().iloc[-1]
    rms = sqrt(mean_squared_error(test['Count'], mov_av['Count']))
    if vis:
        plt.figure(figsize=(8,4))
        plt.plot(train['Count'], label='Train')
        plt.plot(test['Count'], label='Test')
        plt.plot(mov_av['Count'], label='Moving Average Forecast')
        plt.legend(loc='best')
        plt.xlabel('Datapoints')
        plt.ylabel('Signal')
        plt.title(signal)
        plt.show()
    return rms


def best_pred(data, signal, window, vis):
    best_rms = 10000
    best_window = 0
    for i in range(1,window):
        rms = predict(data, signal, False, i)
        if rms < best_rms:
            best_window = i
            best_rms = rms
    print('Best window: %i, with RMSE = %f' % (best_rms, best_rms))
    if vis:
        print(predict(data, signal, True, best_window))


#visualize(data)
# correlate(data)
# plot_corrs(data, ['F_PU1', 'F_PU2', 'S_PU2'])
print(best_pred(data, 'F_PU1', 160, True))
