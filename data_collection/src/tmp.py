from PacpHandler import PcapHandler
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import pylab
import pandas as pd
import dateutil
import datetime
import numpy as np
import csv

def durations(base_dir, timestamps, label):
    input_dir = os.path.join(base_dir, label)
    for root, dirs, files in os.walk(input_dir, topdown=True):
        for name in files:
            # print(os.path.join(root, name))
            if str(name).endswith('.pcap'):
                pcap_path = os.path.join(root, name)
                min_val, max_val = PcapHandler.duration_pcap(pcap_path)
                data = dict()
                data['min'] = datetime.datetime.utcfromtimestamp(min_val)
                data['max'] = datetime.datetime.utcfromtimestamp(max_val)
                data['category'] = label
                timestamps.append(data)


def event_duration(base_dir):
    csv_file = os.path.join(base_dir, 'timestamps.csv')

    if not os.path.exists(csv_file):
        timestamps = []
        durations(base_dir, timestamps, 'TCP-CC')
        durations(base_dir, timestamps, 'Ad')
        with open(csv_file, "wb") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["Category", "Min", "Max"])

            for data in timestamps:
                writer.writerow([data['category'], data['min'], data['max']])

    else:
        df = pd.read_csv(csv_file)

        # map categories to y-values
        cat_dict = dict(zip(pd.unique(df.Category), range(1, len(df.Category) + 1)))

        # map y-values to categories
        val_dict = dict(zip(range(1, len(df.Category) + 1), pd.unique(df.Category)))

        # determing the y-values from categories
        df.Category = df.Category.apply(cat_dict.get)

        df.Min = pd.to_datetime(df.Min).astype(datetime.datetime)
        print df.Min, dt.date2num(df.Min.astype(datetime.datetime))
        df.Max = pd.to_datetime(df.Max).astype(datetime.datetime)
        print dt.date2num(df.Min)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = ax.xaxis_date()
        ax = plt.hlines(df.Category, dt.date2num(df.Min), dt.date2num(df.Max))

        pylab.show()


event_duration('/mnt/Documents/flows/Event/147-32-84-165/')
#event_duration(
 #   '/mnt/Documents/FlowIntent/output/test/2421536307fd9a885cc66c58419cea2e307620dfb67ab96f11aa33380da14c93/')
