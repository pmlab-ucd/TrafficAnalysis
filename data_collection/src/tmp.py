from PacpHandler import PcapHandler
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import json

def event_duration(out_dir):
        timestamps = []
        mins = []
        maxs = []
        index = []
        for root, dirs, files in os.walk(out_dir, topdown=True):
            for name in files:
                # print(os.path.join(root, name))
                if str(name).endswith('.pcap'):
                    pcap_path = os.path.join(root, name)
                    min, max = PcapHandler.duration_pcap(pcap_path)
                    mins.append(min)
                    maxs.append(max)
                    index.append(1)
        data = dict()
        data['index'] = index
        data['mins'] = mins
        data['maxs'] = maxs
        json.dumps(data, os.path.join(out_dir, 'timestamps.json'))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax = ax.xaxis_date()
        ax = plt.hlines(index, mins, maxs)
        fig.show()

#event_duration('/mnt/Documents/flows/Event/147-32-84-165/TCP-CC')
event_duration('/mnt/Documents/FlowIntent/output/test/2421536307fd9a885cc66c58419cea2e307620dfb67ab96f11aa33380da14c93/')