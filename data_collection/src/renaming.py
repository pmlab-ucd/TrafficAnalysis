import os
import re

user = 'hfu'
apk_dir = 'C:\Users\\' + user + '\Documents\FlowIntent\\apks\\drebin\\SerBG\\'
for root, dirs, files in os.walk(apk_dir, topdown=False):
    for filename in files:
        if re.search('apk$', filename):
            continue
        os.rename(os.path.join(root, filename), os.path.join(root, filename + '.apk'))
