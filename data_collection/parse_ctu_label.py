import csv
from scapy.layers.inet import IP, TCP
from scapy.all import *

def filter_pcap(dirname, pcap, time, port, ip):
    try:
        rdpcap(dirname + '/' + ip + '_' + time + '_filtered_port_num_' + str(port) + '.pcap')
        return
    except:
        pass
    try:
        pkts = rdpcap(dirname + '/' + pcap)
    except IOError as e:
        print e.args
        return
    filtered = (pkt for pkt in pkts if
                TCP in pkt
                and (pkt[TCP].sport == port or pkt[TCP].dport == port)
                and (pkt[IP].dst == ip or pkt[IP].src == ip))
    wrpcap(dirname + '/' + ip + '_' + time + '_filtered_port_num_' + str(port) + '.pcap', filtered)
    # for pkt in pkts:
    #     if TCP in pkt:
    #         if pkt[TCP].sport == port or pkt[TCP].dport == port:
    #             print 'Found'

    #print 'done'


def csv_filter_http_c2(item=None, items=None):
    if item['Proto'] != 'tcp':
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
            csv_filter(item, items)
    # print(columns)
    print(items)


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
    read_csv('C:\Users\hfu\Documents\\flows\CTU-13\CTU-13-5\\0\\capture20110815-2.binetflow.2format', csv_filter_http_c2())