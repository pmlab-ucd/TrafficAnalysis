#!/usr/bin/env python2
#-*-encoding:utf-8-*-

from ui_exerciser import UIExerciser
from utils import Utilities
import re
from uiautomator import Device
import time

def pass_first_page(dev):
    for i in range(8):
        time.sleep(1)
        try:
            xml = dev.dump()
        except:
            return
        scroll = re.findall(r'.*?scrollable=\"true\".*?', xml)
        all_text = re.findall(r'.*?text=.*?', xml)
        none_text = re.findall(r'.*?text=\"\".*?', xml)
        if len(scroll) == 1 and (len(all_text) - len(none_text)) <= 2:
            # scroll = re.search(r'.*?scrollable=\"true\".*?bounds=\"(.*?)\"', xml)
            # dev.swipe(400, 0, 0, 0) # for 480 * 800
            dev.swipe(576, 473, 115, 473, 10)
        else:
            break
    # time.sleep(10)
    try:
        xml = dev.dump()
    except:
        return
    clickable = re.findall(r'.*?clickable=\"true\".*?bounds=\"(.*?)\"', xml)
    if len(clickable) == 1:
        node_bounds = clickable[0]
        UIExerciser.touch(dev, node_bounds)
        print 'click single'
    # if detect update info, if 取消， 否
    option_cancle = [u'否', u'取消', u'不升级', u'稍后再说', u'稍后', u'以后'
                                                          u'稍后更新', u'不更新', u'以后再说',
                     u'Not now', u'Cancel', u'以后更新']
    for i in range(5):
        time.sleep(2)
        try:
            xml = dev.dump()
        except:
            return
        clickable = re.findall(r'.*?clickable=\"true\".*?bounds=\"(.*?)\"', xml)
        if len(clickable) <= 3:
            print 'found two clickables'
            # re.findall(r'.*?text=\"(.*?)\".*?[^(text=)].*?clickable=\"true\".*?', xml)
            nodelist = xml.split('><')
            for line in nodelist:
                if re.search('.*?clickable=\"t.*?', line):
                    texts = re.findall(r'text="(.*?)"', line)
                    print texts
                    for text in texts:
                        if text in option_cancle:
                            clickable = re.findall(r'bounds=\"(.*?)\"', line)[0]
                            node_bounds = clickable
                            UIExerciser.touch(dev, node_bounds)
                            print 'click cancle'
                        else:
                            break


if __name__ == '__main__':
    ISOTIMEFORMAT = '%m%d-%H-%M-%S'
    logger = Utilities.set_logger('COSMOS_TRIGGER_PY-Console')

    dev = Device('39302E8CEA9B00EC')
    dev.screen.on()
    pass_first_page(dev)
