#!/usr/bin/env python2
# -*-encoding:utf-8-*-

from uiautomator import Device
import os



if __name__ == '__main__':
    statinfo = os.stat('C:\Users\majes\PycharmProjects\TrafficAnalysis'
                       '\data_collection\output\AnserverBot'
                       '\\30441d6f70304fdefb84d983a99a99a1007d25de\\com.keji.danti6000708-11-39-28.pcap')
    print statinfo.st_size