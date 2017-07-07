#!/usr/bin/env python2
# -*-encoding:utf-8-*-

from ui_exerciser import UIExerciser
from utils import Utilities
import re

import time
from xml.dom.minidom import parseString
from view_client_handler import ViewClientHandler
from subprocess32 import STDOUT, check_output, Popen, PIPE
import os

def adb_kill(name):
    Utilities.logger.debug('Kill: ' + name)
    output = check_output('adb shell ps', stderr=STDOUT, timeout=seconds)
    targets = []
    for line in output.split('\n'):
        #print line
        tmp = line.replace(' ', '')
        tmp = tmp.replace('\n', '')
        if tmp != '':
            # print line
            items = str(line).split(' ')
            items = filter(None, items)
            if name in items[len(items) - 1]:
                targets.append(items[1])
            #Utilities.logger.debug(line)
    for target in targets:
        os.popen('adb shell kill ' + target)


if __name__ == '__main__':
    adb_kill('com.lexa.fakegps')
