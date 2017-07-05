#!/usr/bin/env python2
#-*-encoding:utf-8-*-

from ui_exerciser import UIExerciser
from utils import Utilities
import re
from uiautomator import Device
import time
from xml.dom.minidom import parseString

def pass_first_page(dev):
    for i in range(8):
        xml_data = dev.dump()
        dom = parseString(xml_data.encode("utf-8"))
        nodes = dom.getElementsByTagName('node')
        # Iterate over all the uses-permission nodes
        stay = False
        for node in nodes:
            print node.getAttribute('scrollable'), node.getAttribute('class')
            if node.getAttribute('scrollable') == 'true':
                dev(className=node.getAttribute('class'), scrollable='true').swipe.left()
                stay = True
                break
        if not stay:
            break

    xml_data = dev.dump()
    dom = parseString(xml_data.encode("utf-8"))
    nodes = dom.getElementsByTagName('node')
    # Iterate over all the uses-permission nodes
    stay = False
    clickables = []
    for node in nodes:
        print node.getAttribute('scrollable'), node.getAttribute('class')
        if node.getAttribute('clickable') == 'true':
            clickables.append(node)
    print len(clickables)
    if len(clickables) == 1:
        node_bounds = clickables[0].getAttribute('bounds')
        UIExerciser.touch(dev, node_bounds)
        print 'click single'
    elif len(clickables) == 2:
        # if detect update info, if 取消， 否
        option_cancel = [u'否', u'取消', u'不升级', u'稍后再说', u'稍后', u'以后'
                                                              u'稍后更新', u'不更新', u'以后再说',
                         u'Not now', u'Cancel', u'以后更新', u'取 消']
        for clickable in clickables:
            if clickable.getAttribute('text') in option_cancel:
                UIExerciser.touch(dev, clickable.getAttribute('bounds'))


if __name__ == '__main__':
    ISOTIMEFORMAT = '%m%d-%H-%M-%S'
    logger = Utilities.set_logger('COSMOS_TRIGGER_PY-Console')

    dev = Device('39302E8CEA9B00EC')
    dev.screen.on()
    pass_first_page(dev)
