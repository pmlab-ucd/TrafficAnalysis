Data collection:
0. Crawling Apps

1. Collect pcaps and meta-data (i.e. user interfaces)
1.0 Compile TaintDroid or flash SecDroid (https://app.box.com/s/e4lk7w0gwjl8lsyuzeyr/folder/664963857) on an Android device or an emulator
1.1 Disable unlock mechanism of the device
1.2 Deploy tcpdump and fakegps:
adb push data_collection\tcpdump /data/local/tcpdump
adb shell chmod 6755 /data/local/tcpdump
Set any mock location with fakegps
1.3 Run ui_exerciser_lite to automatically execute apps

