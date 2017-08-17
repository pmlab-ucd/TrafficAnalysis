#!/usr/bin/env python
import json
import os
import re
from PacpHandler import PcapHandler
import dpkt

'''
Process the generated logs from TaintDroid after the execution of the apps
'''


class TaintDroidLogProcessor():
    @staticmethod
    def parse_exerciser_log(log_file):
        with open(log_file) as lines:
            for line in lines:
                if 'pkg:' in line:
                    return line.split('pkg:')[1].replace('\n', '')

    @staticmethod
    def parse_json_log(log_file, pkg):
        '''
        Parse the TaindDroid logs (jsons) and return a tainted record
        :param log_file:
        :param pkg:
        :return:
        '''
        res = []
        try:
            with open(log_file) as data_file:
                taints = json.load(data_file)
                for taint in taints:
                    if taint['process_name'] in pkg:
                        res.append(taint)
            return res
        except Exception as e:
            print e

    @staticmethod
    def filter_pcap_helper(ip, data, packet):
        # Set the TCP data
        tcp = packet.data

        src_ip = PcapHandler.inet_to_str(packet.src)
        dst_ip = PcapHandler.inet_to_str(packet.dst)
        #sport = packet.data.sport
        #dport = packet.data.dport

        if src_ip == ip or dst_ip == ip:
            # print 'Found: ' + dst_ip
            try:
                request = dpkt.http.Request(tcp.data)
                data = str(data).replace('[', '')
                data = data.replace(']', '')

                if 'GET ' in data:
                    data = data.replace('GET ', '')
                elif 'POST ' in data:
                    data = data.replace('POST ', '')
                data = data.replace(' ', '')
                if data in request.uri:
                    return True
                else:
                    # print 'Not matched: ' + ip + ', ' + data + ', ' + request.uri
                    return False
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                return False
        return False

    @staticmethod
    def filter_pcap(args, packet):
        ip = args[0]
        data = args[1]
        # print 'called'
        if TaintDroidLogProcessor.filter_pcap_helper(ip, data, packet):
                return True
        return False

    @staticmethod
    def gen_tag(src):
        src = str(src)
        tag = ''
        if 'Location' in src:
            tag += 'Location_'
        if 'IMEI' in src:
            tag += 'IMEI_'
        if 'ICCID' in src:
            tag += 'ICCID_'
        if 'ContactsProvider' in src:
            tag += 'Address_'
        if 'Microphone Input' in src:
            tag += 'microphone_'
        if 'accelerometer' in src:
            tag += 'accelerometer_'
        if 'camera' in src:
            tag += 'camera'
        return tag

    @staticmethod
    def extract_flow_pcap_helper(taint, pcap_path):
        '''
        Given a taint record, extract the flow in the pcap file and output the pcap flow
        :param taint:
        :return:
        '''
        ip = taint['dst']
        if 'data=' in taint['message']:
            data = taint['message'].split('data=')[1]
        elif 'data' in taint['message']:
            data = taint['message'].split('data')[1]
        else:
            raise Exception
        try:
            return PcapHandler.match_http_requests(pcap_path, TaintDroidLogProcessor.filter_pcap, [ip, data],
                                                   gen_pcap=True, tag=TaintDroidLogProcessor.gen_tag(taint['src']))
        except:
            return []

    @staticmethod
    def extract_flow_pcap(taint, sub_dir):
        flows = []
        for root, dirs, files in os.walk(sub_dir, topdown=False):
            for filename in files:
                if 'filter' not in filename and re.search('pcap$', filename):
                    flows += TaintDroidLogProcessor.extract_flow_pcap_helper(taint, os.path.join(root, filename))
        return flows

    @staticmethod
    def paser_logs(sub_dir):
        pkg = TaintDroidLogProcessor.parse_exerciser_log(sub_dir + '/UIExerciser_FlowIntent_FP_PY.log')
        if pkg:
            print pkg
            for root, dirs, files in os.walk(sub_dir, topdown=False):
                for filename in files:
                    if re.search('json$', filename):
                       return TaintDroidLogProcessor.parse_json_log(os.path.join(root, filename), pkg)

    @staticmethod
    def parse_dir(out_dir):
        flows = {}
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for dir in dirs:
                print os.path.join(root, dir)
                taints = TaintDroidLogProcessor.paser_logs(os.path.join(root, dir))
                if taints:
                    for taint in taints:
                        if 'HTTP' in taint['channel']:
                            print taint
                            flows[str(taint)] = TaintDroidLogProcessor.extract_flow_pcap(taint, os.path.join(root, dir))
        return flows

if __name__ == '__main__':
    taints = TaintDroidLogProcessor.parse_dir('/mnt/Documents/FlowIntent/output/drebin/')
    for taint in taints:
        print taint, taints[taint]
