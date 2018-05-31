import csv
from datetime import datetime


def preprocess(csvfile):
    """Reads the specified CSV file and transforms dates to datetime objects, and strings to floats"""
    datadict = {'date': [], 'L_T1': [], 'L_T2': [], 'L_T3': [], 'L_T4': [], 'L_T5': [], 'L_T6': [], 'L_T7': [],
                'F_PU1': [], 'S_PU1': [], 'F_PU2': [], 'S_PU2': [], 'F_PU3': [], 'S_PU3': [], 'F_PU4': [], 'S_PU4': [],
                'F_PU5': [], 'S_PU5': [], 'F_PU6': [], 'S_PU6': [], 'F_PU7': [], 'S_PU7': [], 'F_PU8': [], 'S_PU8': [],
                'F_PU9': [], 'S_PU9': [], 'F_PU10': [], 'S_PU10': [], 'F_PU11': [], 'S_PU11': [], 'F_V2': [],
                'S_V2': [], 'P_J280': [], 'P_J269': [], 'P_J300': [], 'P_J256': [], 'P_J289': [], 'P_J415': [],
                'P_J302': [], 'P_J306': [], 'P_J307': [], 'P_J317': [], 'P_J14': [], 'P_J422': [], 'flag': []}
    with open(csvfile) as datafile:
        next(datafile)
        reader = csv.reader(datafile)
        keys = list(datadict.keys())
        for row in reader:
            date_string = row[0]
            # Convert date string to datetime object
            date = datetime.strptime(date_string, '%d/%m/%y %H')
            datadict['date'].append(date)
            for i in range(1,len(keys)):
                if datadict[keys[i]] == []:
                    datadict[keys[i]] = [float(row[i])]
                else:
                    datadict[keys[i]].append(float(row[i]))
    return datadict


preprocess('Data/BATADAL_train1.csv')

