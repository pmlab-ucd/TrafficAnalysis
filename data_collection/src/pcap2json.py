#!/usr/bin/env python
"""
    Convert pcap files into jsons, each trace has a json
    Filter traces based on the given labels
"""
import dpkt
import datetime
import socket
from dpkt.compat import compat_ord
# import win_inet_pton
import json
import os
import simplejson
import re

from PacpHandler import PcapHandler
from parse_ctu_label import read_csv, csv_filter_http


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def filter_pcap_helper(ip, port, packet):
    src_ip = PcapHandler.inet_to_str(packet.src)
    dst_ip = PcapHandler.inet_to_str(packet.dst)
    sport = packet.data.sport
    dport = packet.data.dport

    if str(sport) == str(port) or str(dport) == str(port) \
            and src_ip == ip or dst_ip == ip:
        # print 'Found: ' + dst_ip
        return True
    return False


def filter_pcap(args, packet):
    csv_packets = args[0]
    # print 'called'
    for csv_packet in csv_packets:
        if filter_pcap_helper(csv_packet['DstAddr'], csv_packet['Sport'], packet):
            return True
    return False


def filter_flow(args, flow):
    csv_packets = args[0]

    # print 'called'
    for csv_packet in csv_packets:
        dest_ip = csv_packet['DstAddr']
        sport = csv_packet['Sport']
        time = csv_packet['StartTime']
        if str(sport) == str(flow['sport']) and str(flow['dest']) == str(dest_ip):
            # if time == flow['timestamp']:
            # print 'Found: ' + dst_ip
            return True
            # else:
            #   print time, flow['timestamp']

    return False


def pcap2jsons(pcap_dir, out_dir, filter_func=None, *args):
    filtered = []
    for root, dirs, files in os.walk(pcap_dir, topdown=True):
        for name in files:
            # print(os.path.join(root, name))
            if str(name).endswith('.pcap'):
                pcap_path = os.path.join(root, name)
                if '/0/' in pcap_path:
                    label = 0
                elif '/1/' in pcap_path:
                    label = 1
                else:
                    label = 'unknown'
                """Open up a test pcap file and print out the packets"""
                print pcap_path
                filtered += PcapHandler.http_requests(pcap_path, filter_flow=filter_func, args=args)
    print out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for flow in filtered:
        timestamp = flow['timestamp'].replace(':', '-')
        timestamp = timestamp.replace('.', '-')
        timestamp = timestamp.replace(' ', '_')
        filename = str(flow['domain'] + '_' + timestamp + '.json').replace(':', '_').replace('/', '_')
        with open(os.path.join(out_dir, filename), 'w') as outfile:
            try:
                json.dump(flow, outfile)
            except UnicodeDecodeError as e:
                print e
    return filtered


def dir2jsons(json_dir):
        jsons = []
        if json_dir is None:
            return jsons
        for root, dirs, files in os.walk(json_dir, topdown=False):
            for filename in files:
                if '201' in filename and re.search('json$', filename):
                    with open(os.path.join(root, filename), "rb") as fin:
                        try:
                            jsons.append(simplejson.load(fin))
                        except Exception as e:
                            pass
                            # Utilities.logger.error(e)
        return jsons


def json2pcap(json_dir, pcap_path, out_dir, tag=''):
    """
    Parse json and filter the pcap to generte subpcap
    :param json_dir:
    :param pcap_path:
    :param out_dir:
    :return:
    """
    jsons = dir2jsons(json_dir)
    print len(jsons)
    pkts = PcapHandler.get_packets(pcap_path)
    for json in jsons:
        print json
        PcapHandler.filter_pcap(out_dir, pkts, json['dest'],
                            json['sport'], tag=tag)


if __name__ == '__main__':
    '''
    dir = '/mnt/Documents/flows/CTU-13/CTU-13-1/'
    dir_0 = dir + '0/'
    label = 'Ad' #TCP-CC' #SPAM'
    for root, dirs, files in os.walk(dir_0, topdown=True):
        for name in files:
            # print(os.path.join(root, name))
            if str(name).endswith('binetflow.2format'):
                headers, csv_packets = read_csv(os.path.join(dir_0, name),
                                    csv_filter_http, label=label)
                flows = pcap2jsons(dir_0, dir_0 + '/' + label, filter_flow, csv_packets)
                print len(csv_packets), len(flows)

                for csv_packet in csv_packets:
                    print csv_packet['DstAddr'], csv_packet['Sport'], csv_packet['StartTime'], csv_packet['LastTime']
                print '_________________________________'
                for flow in flows:
                    print flow['dest'], flow['sport'], flow['uri']

    '''
    #dir_1 = '/mnt/Documents/flows/FlowIntent/Address'
    #pcap2jsons(dir_1, dir_1)

    json2pcap('/mnt/Documents/flows/Event/TCP-CC', '/mnt/Documents/flows/Event/botnet-capture-20110817-bot.pcap',
              '/mnt/Documents/flows/Event/TCP-CC', tag='TCP-CC')
