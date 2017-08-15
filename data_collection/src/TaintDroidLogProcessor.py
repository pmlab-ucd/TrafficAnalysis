#!/usr/bin/env python
import json
import os
import re

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
        try:
            with open(log_file) as data_file:
                taints = json.load(data_file)
                for taint in taints:
                    if taint['process_name'] == pkg:
                        print taint
        except Exception as e:
            print e

    @staticmethod
    def paser_logs(sub_dir):
        pkg = TaintDroidLogProcessor.parse_exerciser_log(sub_dir + '\UIExerciser_FlowIntent_FP_PY.log')
        if pkg:
            for root, dirs, files in os.walk(sub_dir, topdown=False):
                for filename in files:
                    if re.search('json$', filename):
                        TaintDroidLogProcessor.parse_json_log(os.path.join(root, filename), pkg)

    @staticmethod
    def parse_dir(out_dir):
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for dir in dirs:
                print dir
                print os.path.join(root, dir)
                TaintDroidLogProcessor.paser_logs(os.path.join(root, dir))

if __name__ == '__main__':
    TaintDroidLogProcessor.parse_dir('C:\Users\hfu\PycharmProjects\TrafficAnalysis\data_collection\output'
                                                '\DroidKungFu')
