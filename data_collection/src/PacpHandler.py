__author__ = 'hao'

from scapy.layers.inet import IP, TCP
from scapy.all import *
import dpkt
import datetime
import socket
from dpkt.compat import compat_ord

# import win_inet_pton

'''
 match flow in pcap to the flow reported by TaintDroid
'''


class PcapHandler:
    @staticmethod
    def filter_pcap_by_ip(dirname, pkts, ip):
        ip = str(ip)
        output_path = os.path.join(dirname, ip + '.pcap')
        if os.path.exists(output_path):
            return

        filtered = []
        for pkt in pkts:
            if TCP in pkt:
                # print pkt[TCP].sport
                if pkt[IP].dst == ip or pkt[IP].src == ip:
                    #print 'Found: ' + pkt[IP].dst
                    filtered.append(pkt)

        # filtered = (pkt for pkt in pkts if
        #            TCP in pkt
        #            and (str(pkt[TCP].sport) == port or str(pkt[TCP].dport) == port)
        #            and (pkt[IP].dst == ip or pkt[IP].src == ip))
        wrpcap(output_path, filtered)

    @staticmethod
    def filter_pcap(dirname, pkts, ip, port, tag=''):
        ip = str(ip)
        port = str(port)
        output_path = os.path.join(dirname, ip + '_filtered_' + tag + '_' + str(port) + '.pcap')
        if os.path.exists(output_path):
            return

        filtered = []
        for pkt in pkts:
            if TCP in pkt:
                # print pkt[TCP].sport
                if str(pkt[TCP].sport) == str(port) or str(pkt[TCP].dport) == str(port) \
                        and pkt[IP].dst == ip or pkt[IP].src == ip:
                    # print 'Found: ' + pkt[IP].dst
                    filtered.append(pkt)

        # filtered = (pkt for pkt in pkts if
        #            TCP in pkt
        #            and (str(pkt[TCP].sport) == port or str(pkt[TCP].dport) == port)
        #            and (pkt[IP].dst == ip or pkt[IP].src == ip))
        wrpcap(output_path, filtered)

    @staticmethod
    def http_requests(pcap_path, label='', filter_func=None, filter_flow=None, args=None):
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            # flows = print_http_requests(pcap, label, filter_func, args)
            return PcapHandler.http_requests_helper(pcap, label, filter_func=filter_func,
                                                    filter_flow=filter_flow, args=args)
            # out_dir = os.curdir + '/output/' + os.path.basename(os.path.abspath(os.path.join(root, os.pardir))) + '/' + str(
            #   label) + '/'

    @staticmethod
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

    @staticmethod
    def http_requests_helper(pcap, label='', filter_func=None, filter_flow=None, args=None):
        """Print out information about each packet in a pcap

               Args:
                   pcap: dpkt pcap reader object (dpkt.pcap.Reader)
            """
        # For each packet in the pcap process the contents
        flows = []
        examined = []  # Do not know why there are redundant flows
        for timestamp, buf in pcap:

            # Unpack the Ethernet frame (mac src/dst, ethertype)
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except:
                continue
            # Make sure the Ethernet data contains an IP packet
            if not isinstance(eth.data, dpkt.ip.IP):
                # print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
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

                if filter_func is not None and not filter_func(args, packet):
                        continue
                flow = dict()
                # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
                do_not_fragment = bool(packet.off & dpkt.ip.IP_DF)
                more_fragments = bool(packet.off & dpkt.ip.IP_MF)
                fragment_offset = packet.off & dpkt.ip.IP_OFFMASK

                # Print out the info
                timestamp = str(datetime.datetime.utcfromtimestamp(timestamp))

                # print('Timestamp: ', timestamp)
                # print('Ethernet Frame: ', mac_addr(eth.src), mac_addr(eth.dst), eth.type)
                # print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' %
                #      (inet_to_str(packet.src), inet_to_str(packet.dst), packet.len, packet.ttl, do_not_fragment,
                #       more_fragments, fragment_offset))
                # print('HTTP request: %s\n' % repr(request))
                # print tcp.sport, tcp.dport
                flow['label'] = label
                flow['post_body'] = request.body
                try:
                    flow['domain'] = request.headers['host']
                except:
                    flow['domain'] = str(PcapHandler.inet_to_str(packet.dst))
                flow['uri'] = request.uri
                flow['headers'] = request.headers
                flow['platform'] = 'unknown'
                flow['referrer'] = 'unknown'
                flow['src'] = PcapHandler.inet_to_str(packet.src)
                flow['sport'] = tcp.sport
                flow['dest'] = PcapHandler.inet_to_str(packet.dst)
                flow['dport'] = tcp.dport
                flow['request'] = repr(request)
                flow['timestamp'] = timestamp
                # print repr(flow)
                id = flow['dest'] + str(flow['sport']) + flow['uri']
                if filter_flow is not None and filter_flow(args, flow):
                    continue

                if id not in examined:
                    flows.append(flow)
                    examined.append(id)

                # Check for Header spanning acrossed TCP segments
                if not tcp.data.endswith(b'\r\n'):
                    # print('\nHEADER TRUNCATED! Reassemble TCP segments!\n')
                    pass
        return flows

    @staticmethod
    def mac_addr(address):
        """Convert a MAC address to a readable/printable string

           Args:
               address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
           Returns:
               str: Printable/readable MAC address
        """
        return ':'.join('%02x' % compat_ord(b) for b in address)

    @staticmethod
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


    @staticmethod
    def match_http_requests(pcap_path, filter_func, args, gen_pcap=False, tag=''):
        '''
        Match requests based filter_func
        :param pcap:
        :param label:
        :param filter_func:
        :param args:
        :return: flows
        '''
        # For each packet in the pcap process the contents
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            flows = []
            debug = False
            print pcap_path
            for timestamp, buf in pcap:
                time_stamp = timestamp

                # Unpack the Ethernet frame (mac src/dst, ethertype)
                eth = dpkt.ethernet.Ethernet(buf)

                # Make sure the Ethernet data contains an IP packet
                if not isinstance(eth.data, dpkt.ip.IP):
                    # print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
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
                    if debug:
                        print('Timestamp: ', timestamp)
                        print(
                        'Ethernet Frame: ', PcapHandler.mac_addr(eth.src), PcapHandler.mac_addr(eth.dst), eth.type)
                        print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' %
                              (PcapHandler.inet_to_str(packet.src), PcapHandler.inet_to_str(packet.dst), packet.len,
                               packet.ttl, do_not_fragment,
                               more_fragments, fragment_offset))
                        print('HTTP request: %s\n' % repr(request))
                        print tcp.sport, tcp.dport
                    flow['post_body'] = request.body
                    try:
                        flow['domain'] = request.headers['host']
                    except:
                        flow['domain'] = str(PcapHandler.inet_to_str(packet.dst))
                    flow['uri'] = request.uri
                    flow['headers'] = request.headers
                    flow['platform'] = 'unknown'
                    flow['referrer'] = 'unknown'
                    timestamp = timestamp.replace(':', '-')
                    timestamp = timestamp.replace('.', '-')
                    timestamp = timestamp.replace(' ', '_')
                    flow['timestamp'] = timestamp
                    if debug:
                        print repr(flow)
                    flows.append(flow)

                    # Check for Header spanning acrossed TCP segments
                    if not tcp.data.endswith(b'\r\n'):
                        print('\nHEADER TRUNCATED! Reassemble TCP segments!\n')

                    if gen_pcap:
                        '''
                        # The following way does not count http response
                        file = open('file.pcap', 'wb')
                        pcapfile = dpkt.pcap.Writer(file)
                        pcapfile.writepkt(buf, time_stamp)
                        pcapfile.close()
                        file.close()
                        '''
                        pkts = PcapHandler.get_packets(pcap_path)
                        PcapHandler.filter_pcap(os.path.dirname(pcap_path), pkts, PcapHandler.inet_to_str(packet.dst),
                                                tcp.sport, tag=tag)

            return flows


    @staticmethod
    def print_pacp(pcap):
        # For each packet in the pcap process the contents
        for timestamp, buf in pcap:

            # Unpack the Ethernet frame (mac src/dst, ethertype)
            eth = dpkt.ethernet.Ethernet(buf)

            # Make sure the Ethernet data contains an IP packet
            if not isinstance(eth.data, dpkt.ip.IP):
                print 'Non IP Packet type not supported %s\n' % eth.data.__class__.__name__
                continue

            # Now grab the data within the Ethernet frame (the IP packet)
            ip = eth.data

            # Check for TCP in the transport layer
            if isinstance(ip.data, dpkt.tcp.TCP):

                # Set the TCP data
                tcp = ip.data

                # Now see if we can parse the contents as a HTTP request
                try:
                    request = dpkt.http.Request(tcp.data)
                except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                    continue

                # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
                do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
                more_fragments = bool(ip.off & dpkt.ip.IP_MF)
                fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

                # Print out the info
                print 'Timestamp: ', str(datetime.datetime.utcfromtimestamp(timestamp))
                print 'Ethernet Frame: ', PcapHandler.mac_addr(eth.src), PcapHandler.mac_addr(eth.dst), eth.type
                print 'IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' % \
                      (
                      PcapHandler.inet_to_str(ip.src), PcapHandler.inet_to_str(ip.dst), ip.len, ip.ttl, do_not_fragment,
                      more_fragments,
                      fragment_offset)
                print 'HTTP request: %s\n' % repr(request)

    @staticmethod
    def get_packets(pcap_path):
        try:
            pkts = rdpcap(pcap_path)
            return pkts
        except IOError as e:
            print e.args
            return

if __name__ == '__main__':
    pcap_path = '/mnt/Documents/FlowIntent/output/test/cfa9c6ea949a3fc002c38ca3510acfbb5ec5a210d56c12ecac815b8806718602/com.appspot.swisscodemonkeys.steam0711-21-34-02.pcap'
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        PcapHandler.print_pacp(pcap)
