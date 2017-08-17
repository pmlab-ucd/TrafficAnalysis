import csv

from scapy.layers.inet import IP, TCP
from scapy.all import *
from PacpHandler import PcapHandler





def csv_filter_http_c2(item=None, items=None):
    if item['Proto'] != 'tcp' or '-TCP-CC' not in item['Label']:
        return
    items.append(item)
    print item


def read_csv(csv_path, csv_filter):
    f = open(csv_path)
    reader = csv.reader(f)
    headers = next(reader, None)

    items = []
    for row in reader:
        # print row
        item = dict()
        for h, v in zip(headers, row):
            item[h] = v
            # columns[h].append(v)
        csv_filter(item=item, items=items)
    # print(columns)
    print(len(items))

    columns = {}
    for h in headers:
        columns[h] = []

    for row in reader:
        for h, v in zip(headers, row):
            columns[h].append(v)
    print(len(columns))

    headers = []
    for h in columns:
        if len(str(h)) == 0:
            continue
        headers.append(h)

    return headers, items


if __name__ == '__main__':
    headers, packets = read_csv('/mnt/Documents/flows/CTU-13/CTU-13-1/0/capture20110810.binetflow.2format',
                                csv_filter_http_c2)

    dirname = '/mnt/Documents/flows/CTU-13/CTU-13-1/0/'
    pcap = 'botnet-capture-20110810-neris.pcap'
    pkts = PcapHandler.get_packets(os.path.join(dirname, pcap))
    for packet in packets:
        PcapHandler.filter_pcap(dirname, pkts, packet['DstAddr'], packet['Sport'])
