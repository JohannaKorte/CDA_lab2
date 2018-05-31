import preprocess
import matplotlib.pyplot as plt

filename = 'Data/BATADAL_train1.csv'


def visualize(filename):
    """ Visualizes the behavior of all signals over time. """
    data = preprocess.preprocess(filename)
    for key in data.keys():
        if key == 'data':
            continue
        else:
            plt.plot(data['date'][:1000], data[key][:1000])
            plt.show()

visualize(filename)