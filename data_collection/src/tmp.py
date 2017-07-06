#!/usr/bin/env python2
#-*-encoding:utf-8-*-

from ui_exerciser import UIExerciser
from utils import Utilities
import re
from uiautomator import Device
import time
from xml.dom.minidom import parseString
from view_client_handler import ViewClientHandler


if __name__ == '__main__':
    ViewClientHandler.dump_view_server('com.kuaihuoyun.driver')
