import os

user = 'majes'
apk_dir = 'C:\Users\\' + user + '\Documents\FlowIntent\\apks\\drebin\\DroidKungFu\\'
for root, dirs, files in os.walk(apk_dir, topdown=False):
    for filename in files:
        os.rename(os.path.join(root, filename), os.path.join(root, filename + '.apk'))
