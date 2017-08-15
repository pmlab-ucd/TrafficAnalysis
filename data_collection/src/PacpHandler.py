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
        try:
            rdpcap(dirname + '/' + ip  + '.pcap')
            return
        except:
            pass

        filtered = []
        for pkt in pkts:
            if TCP in pkt:
                # print pkt[TCP].sport
                if pkt[IP].dst == ip or pkt[IP].src == ip:
                    print 'Found: ' + pkt[IP].dst
                    filtered.append(pkt)

        # filtered = (pkt for pkt in pkts if
        #            TCP in pkt
        #            and (str(pkt[TCP].sport) == port or str(pkt[TCP].dport) == port)
        #            and (pkt[IP].dst == ip or pkt[IP].src == ip))
        wrpcap(dirname + '/' + ip +  '.pcap', filtered)


    @staticmethod
    def filter_pcap(dirname, pkts, ip, port):
        ip = str(ip)
        port = str(port)
        try:
            rdpcap(dirname + '/' + ip + '_filtered_port_num_' + str(port) + '.pcap')
            return
        except:
            pass

        print port

        filtered = []
        for pkt in pkts:
            if TCP in pkt:
                # print pkt[TCP].sport
                if str(pkt[TCP].sport) == str(port) or str(pkt[TCP].dport) == str(port) \
                        and pkt[IP].dst == ip or pkt[IP].src == ip:
                    print 'Found: ' + pkt[IP].dst
                    filtered.append(pkt)

        # filtered = (pkt for pkt in pkts if
        #            TCP in pkt
        #            and (str(pkt[TCP].sport) == port or str(pkt[TCP].dport) == port)
        #            and (pkt[IP].dst == ip or pkt[IP].src == ip))
        wrpcap(dirname + '/' + ip + '_filtered_port_num_' + str(port) + '.pcap', filtered)

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
                      (PcapHandler.inet_to_str(ip.src), PcapHandler.inet_to_str(ip.dst), ip.len, ip.ttl, do_not_fragment, more_fragments,
                       fragment_offset)
                print 'HTTP request: %s\n' % repr(request)

if __name__ == '__main__':
    PcapHandler.print_pacp('C:\Users\hfu\PycharmProjects\TrafficAnalysis\data_collection\output'
                                                '\DroidKungFu\\'
                           '2f6dfbd2621805916fe22b565e7c6869d9fa3815d9cc1ebda4573ae384a952b6\\'
                           'com.mogo.puzzle0710-09-01-13.pcap')