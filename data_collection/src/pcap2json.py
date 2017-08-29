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
from parse_ctu_label import read_csv, csv_filter_http


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def filter_pcap_helper(ip, port, packet):
    src_ip = inet_to_str(packet.src)
    dst_ip = inet_to_str(packet.dst)
    sport = packet.data.sport
    dport = packet.data.dport

    if str(sport) == str(port) or str(dport) == str(port) \
            and src_ip == ip or dst_ip == ip:
        #print 'Found: ' + dst_ip
        return True
    return False


def filter_pcap(args, packet):
    csv_packets = args[0]
    #print 'called'
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
            #if time == flow['timestamp']:
            # print 'Found: ' + dst_ip
              return True
            #else:
             #   print time, flow['timestamp']

    return False


def get_all_flows(pcap):
    """Print out information about each packet in a pcap

           Args:
               pcap: dpkt pcap reader object (dpkt.pcap.Reader)
        """
    # For each packet in the pcap process the contents
    flows = []
    examined = [] # Do not know why there are redundant flows
    for timestamp, buf in pcap:

        # Unpack the Ethernet frame (mac src/dst, ethertype)
        eth = dpkt.ethernet.Ethernet(buf)

        # Make sure the Ethernet data contains an IP packet
        if not isinstance(eth.data, dpkt.ip.IP):
            #print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue

        # Now grab the data within the Ethernet frame (the IP packet)
        packet = eth.data

        # Check for TCP in the transport layer
        if isinstance(packet.data, dpkt.tcp.TCP):

            # Set the TCP data
            tcp = packet.data

            # Now see if we can parse the contents as a HTTP request
            try:
                request = dpkt.http.Request(tcp.data)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            flow = dict()
            # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
            do_not_fragment = bool(packet.off & dpkt.ip.IP_DF)
            more_fragments = bool(packet.off & dpkt.ip.IP_MF)
            fragment_offset = packet.off & dpkt.ip.IP_OFFMASK

            # Print out the info
            timestamp = str(datetime.datetime.utcfromtimestamp(timestamp))

            #print('Timestamp: ', timestamp)
            #print('Ethernet Frame: ', mac_addr(eth.src), mac_addr(eth.dst), eth.type)
            #print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' %
            #      (inet_to_str(packet.src), inet_to_str(packet.dst), packet.len, packet.ttl, do_not_fragment,
            #       more_fragments, fragment_offset))
            #print('HTTP request: %s\n' % repr(request))
            #print tcp.sport, tcp.dport
            flow['label'] = label
            flow['post_body'] = request.body
            try:
                flow['domain'] = request.headers['host']
            except:
                flow['domain'] = str(inet_to_str(packet.dst))
            flow['uri'] = request.uri
            flow['headers'] = request.headers
            flow['platform'] = 'unknown'
            flow['referrer'] = 'unknown'
            flow['src'] = inet_to_str(packet.src)
            flow['sport'] = tcp.sport
            flow['dest'] = inet_to_str(packet.dst)
            flow['dport'] = tcp.dport

            flow['timestamp'] = timestamp
            # print repr(flow)
            id = flow['dest'] + str(flow['sport']) + flow['uri']
            if id not in examined:
                flows.append(flow)
                examined.append(id)

            # Check for Header spanning acrossed TCP segments
            if not tcp.data.endswith(b'\r\n'):
                # print('\nHEADER TRUNCATED! Reassemble TCP segments!\n')
                pass
    return flows


def print_http_requests(pcap, label, filter_func, args):
    """Print out information about each packet in a pcap

       Args:
           pcap: dpkt pcap reader object (dpkt.pcap.Reader)
    """
    # For each packet in the pcap process the contents
    flows = []
    for timestamp, buf in pcap:

        # Unpack the Ethernet frame (mac src/dst, ethertype)
        eth = dpkt.ethernet.Ethernet(buf)

        # Make sure the Ethernet data contains an IP packet
        if not isinstance(eth.data, dpkt.ip.IP):
            print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue

        # Now grab the data within the Ethernet frame (the IP packet)
        packet = eth.data

        # Check for TCP in the transport layer
        if isinstance(packet.data, dpkt.tcp.TCP):
            if filter_func and not filter_func(args, packet):
                continue

            # Set the TCP data
            tcp = packet.data

            # Now see if we can parse the contents as a HTTP request
            try:
                request = dpkt.http.Request(tcp.data)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            flow = dict()
            # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
            do_not_fragment = bool(packet.off & dpkt.ip.IP_DF)
            more_fragments = bool(packet.off & dpkt.ip.IP_MF)
            fragment_offset = packet.off & dpkt.ip.IP_OFFMASK

            # Print out the info
            timestamp = str(datetime.datetime.utcfromtimestamp(timestamp))
            print('Timestamp: ', timestamp)
            print('Ethernet Frame: ', mac_addr(eth.src), mac_addr(eth.dst), eth.type)
            print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' %
                  (inet_to_str(packet.src), inet_to_str(packet.dst), packet.len, packet.ttl, do_not_fragment,
                   more_fragments, fragment_offset))
            print('HTTP request: %s\n' % repr(request))
            print tcp.sport, tcp.dport
            flow['label'] = label
            flow['post_body'] = request.body
            try:
                flow['domain'] = request.headers['host']
            except:
                flow['domain'] = str(inet_to_str(packet.dst))
            flow['uri'] = request.uri
            flow['headers'] = request.headers
            flow['platform'] = 'unknown'
            flow['referrer'] = 'unknown'
            timestamp = timestamp.replace(':', '-')
            timestamp = timestamp.replace('.', '-')
            timestamp = timestamp.replace(' ', '_')
            flow['timestamp'] = timestamp
            print repr(flow)
            flows.append(flow)

            # Check for Header spanning acrossed TCP segments
            if not tcp.data.endswith(b'\r\n'):
                print('\nHEADER TRUNCATED! Reassemble TCP segments!\n')
    return flows


def pcap2jsons(pcap_dir, out_dir, filter_func=None, *args):
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
                with open(pcap_path, 'rb') as f:
                    pcap = dpkt.pcap.Reader(f)
                    # flows = print_http_requests(pcap, label, filter_func, args)
                    flows = get_all_flows(pcap)
                    #out_dir = os.curdir + '/output/' + os.path.basename(os.path.abspath(os.path.join(root, os.pardir))) + '/' + str(
                     #   label) + '/'

                    print out_dir
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    #print len(args[0])
                    count = 0
                    filtered = []
                    for flow in flows:
                        count += 1
                        #print flow['dest'], flow['sport']
                        if not filter_flow(args, flow):
                            continue
                        #count += 1
                        timestamp = flow['timestamp'].replace(':', '-')
                        timestamp = timestamp.replace('.', '-')
                        timestamp = timestamp.replace(' ', '_')
                        filename = str(flow['domain'] + '_' + timestamp + '.json').replace(':', '_')
                        filtered.append(flow)
                        with open(out_dir + '/' + filename, 'w') as outfile:
                            try:
                                json.dump(flow, outfile)
                            except UnicodeDecodeError as e:
                                print e
                    #print 'count', count
                    return filtered


if __name__ == '__main__':
    dir = '/mnt/Documents/flows/CTU-13/CTU-13-7/'
    dir_0 = dir + '0/'
    label = 'TCP-CC'
    headers, csv_packets = read_csv(dir_0 + 'capture20110816-2.binetflow.2format',
                                    csv_filter_http, label=label)
    flows = pcap2jsons(dir_0, dir_0 + '/' + label, filter_flow, csv_packets)
    print len(csv_packets), len(flows)

    for csv_packet in csv_packets:
        print csv_packet['DstAddr'], csv_packet['Sport'], csv_packet['StartTime'], csv_packet['LastTime']
    print '_________________________________'
    for flow in flows:
        print flow['dest'], flow['sport'], flow['uri']

    #dir_1 = dir + '1/'
    #pcap2jsons(dir_1)
