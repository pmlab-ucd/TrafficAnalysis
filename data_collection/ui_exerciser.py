from uiautomator import Device
from xml.dom.minidom import parseString
from utils import Utilities
import os
from subprocess32 import STDOUT, check_output, Popen, PIPE
import re
import csv
import time
import psutil
import sign_apks
import json

class UIExerciser:

    emu_proc = None
    emu_loc = None
    emu_name = None
    series = None

    def start_activity(self, package, activity):
        self.logger.info("Start Activity " + activity)
        # cmd = 'adb -s ' + series + ' shell am start -D -n ' + package + '/' + activity
        # os.popen('adb -s ' + series + ' shell am start -n ' + package + '/.' + activity)
        # cmd = self.monkeyrunner_loc + ' ' + os.getcwd() + '/run_activity_monkeyrunner.py ' + ' ' + \
        # self.series + ' ' + package + '/' + activity
        cmd = 'adb -s ' + self.series + ' shell am start -n ' + package + '/' + activity
        return UIExerciser.run_adb_cmd(cmd)

    def get_package_name(self, aapt, apk):
        cmd = aapt + ' dump badging ' + apk
        try:
            output = check_output(cmd, stderr=STDOUT)
        except Exception as e:
            print e
            return None
        for line in output.split('\n'):
            if 'package: name=' in line:
                # the real code does filtering here
                package = re.findall('\'([^\']*)\'', line.rstrip())[0]
                self.logger.info('Package: ' + package)
                return package
            else:
                break

    def get_launchable_activity(self, aapt, apk):
        cmd = aapt + ' dump badging ' + apk
        activities = []
        try:
            output = check_output(cmd, stderr=STDOUT)
        except Exception as e:
            print e
            return None
        for line in output.split('\n'):
            if 'launchable-activity: name=' in line:
                # the real code does filtering here
                activity = re.findall('\'([^\']*)\'', line.rstrip())[0]
                self.logger.info('Launchable activity: ' + activity)
                activities.append(activity)
            else:
                break
        return activities

    def is_crashed(self, dev, xml_data):
        dom = parseString(xml_data.encode("utf-8"))
        nodes = dom.getElementsByTagName('node')
        # Iterate over all the uses-permission nodes
        # crashed = True
        for node in nodes:
            if node.getAttribute('text') != '':
                if ' has stopped.' in node.getAttribute('text'):
                    if dev(resourceId="android:id/button1", text="OK").exists:
                        dev(resourceId="android:id/button1", text="OK").click()
                    self.logger.warn('Crashed!')
                    return True
                    # print(node.getAttribute('text'))
                    # print(node.toxml())
                    # if node.getAttribute('package') == package:
                    # crashed = False
        return False

    def is_SMS_alarm(self, dev, xml_data):
        dom = parseString(xml_data.encode("utf-8"))
        nodes = dom.getElementsByTagName('node')
        # Iterate over all the uses-permission nodes
        # crashed = True
        for node in nodes:
            if node.getAttribute('text') != '':
                if 'would like to send a message to ' in node.getAttribute('text'):
                    if dev(resourceId='android:id/button2', text="Cancel").exists:
                        # print dev.press.back()
                        # print dev(resourceId='android:id/sms_short_code_remember_choice_checkbox').click()
                        print dev(resourceId='android:id/button2', text="Cancel").click()
                        print('Send SMS alarm')
                    return True
        return False

    def screenshot(self, dir_data, activity):
        # current_time = time.strftime(ISOTIMEFORMAT, time.localtime())
        self.logger.info('Try to dump layout XML of ' + activity)
        try:
            dev = Device(self.series)
            dev.screen.on()
            activity = str(activity).replace('\"', '')
            # dev.wait.idle()
            self.logger.info('Dumping...' + activity)
            xml_data = dev.dump(dir_data + activity + '.xml')
            self.logger.info(xml_data)
            self.is_crashed(dev, xml_data)
            while self.is_SMS_alarm(dev, xml_data):
                xml_data = dev.dump(dir_data + activity + '.xml')
            self.logger.info(dev.screenshot(dir_data + activity + '.png'))
            return True
        except Exception as e:
            self.logger.error("Error when screenshot: " + activity + ', due to ' + e.message)
            return False

    def start_activities(self, package, csvpath, output_dir):
        self.logger.info("Try to read csv " + csvpath)
        csvfile = open(csvpath, 'rb')
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            activity = row[0]
            if self.start_activity(package, activity):
                time.sleep(2)
                # self.screenshot(output_dir, activity)
                for i in range(1, 3):
                    if not UIExerciser.check_dev_online(UIExerciser.series):
                        if UIExerciser.emu_proc:
                            UIExerciser.close_emulator(UIExerciser.emu_proc)
                            UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
                        else:
                            raise Exception('Cannot start Activity ' + activity)
                    if Utilities.run_method(self.screenshot, 180, args=[output_dir, activity]) :
                        break
                    else:
                        self.logger.warn("Time out while dumping XML for " + activity)
            else:
                time.sleep(2)
                self.logger.error('Cannot start Activity: ' + activity)



    def __init__(self, series, aapt_loc, trigger_java_dir, apk_dir, monkeyrunner_loc, out_base_dir, logger):
        self.series = series
        UIExerciser.series = series
        self.aapt_loc = aapt_loc
        self.trigger_java_dir = trigger_java_dir
        self.apk_dir = apk_dir
        self.monkeyrunner_loc = monkeyrunner_loc
        self.logger = logger
        self.out_base_dir = out_base_dir

    @staticmethod
    def check_examined(out_dir):
        examined = []
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for name in dirs:
                # print(os.path.join(root, name))
                examined.append(name)

        print len(examined)
        return examined

    @staticmethod
    def get_csv_path(base_dir, par_dir, apk_name):
        path = os.path.join(base_dir, par_dir)
        path = os.path.join(path, apk_name)
        path = os.path.join(path, apk_name + '.apk_tgtAct.csv')
        return path

    @staticmethod
    def open_emulator(emu_loc, emu_name):
        cmd = emu_loc + ' @' + emu_name
        pro = Popen(cmd, stdout=PIPE, shell=True)

        for i in range(1, 5):
            time.sleep(20)
            if UIExerciser.check_dev_online(UIExerciser.series):
                return pro

        return False

    @staticmethod
    def open_emu(emu_loc, emu_name):
        cmd = emu_loc + ' @' + emu_name
        emu_proc = Popen(cmd, stdout=PIPE, shell=True)
        while not UIExerciser.check_dev_online(UIExerciser.series):
            time.sleep(10)
            print 'waiting for the emulator ' + UIExerciser.series
        print emu_name + ' found'
        return emu_proc

    @staticmethod
    def check_dev_online(series):
        try:
            output = check_output('adb devices', stderr=STDOUT, timeout=10)
            for line in output.split('\n'):
                if series in line:
                    if 'device' in line:
                        return True
        except Exception as exp:
            print(exp.message)
            return False

    @staticmethod
    def kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        try:
            process.kill()
        except Exception as e:
            print e.message

    @staticmethod
    def close_emulator(emu_proc):
        UIExerciser.kill(emu_proc.pid)

    @staticmethod
    def install_apk(series, apk):
        cmd = 'install ' + apk
        if not UIExerciser.run_adb_cmd(cmd):
            sign_apks.sign_apk(apk)
            if not UIExerciser.run_adb_cmd(cmd):
                raise Exception('Cannot install ' + apk)

    @staticmethod
    def uninstall_pkg(series, pkg):
        cmd = 'uninstall ' + pkg
        UIExerciser.run_adb_cmd(cmd, series=series)

    @staticmethod
    def run_adb_cmd(cmd, series=None):
        if series:
            return UIExerciser.run_cmd('adb -s ' + UIExerciser.series + ' ' + cmd)
        else:
            return UIExerciser.run_cmd('adb ' + cmd)

    @staticmethod
    def run_cmd(cmd):
        Utilities.logger.debug('Run cmd: ' + cmd)

        seconds = 60
        result = True
        for i in range(1, 3):
            try:
                result = True
                output = check_output(cmd, stderr=STDOUT, timeout=seconds)
                for line in output.split('\n'):
                    if 'Failure' in line or 'Error' in line:
                        result = False
                    tmp = line.replace(' ', '')
                    tmp = tmp.replace('\n', '')
                    if tmp != '':
                        Utilities.logger.debug(line)
                break
            except Exception as exc:
                Utilities.logger.warn(exc)
                result = False
                if not UIExerciser.check_dev_online(UIExerciser.series):
                    if UIExerciser.emu_proc:
                        UIExerciser.close_emulator(UIExerciser.emu_proc)
                        UIExerciser.emu_proc = UIExerciser.open_emu(UIExerciser.emu_loc, UIExerciser.emu_name)
                    else:
                        raise Exception(cmd)

        return result

    @staticmethod
    def start_taintdroid(series):
        UIExerciser.run_adb_cmd('adb -s ' + series + ' shell am start -n fu.hao.uidroid/.TaintDroidNotifyController')

    def flowintent_first_page(self, series, apk, examined):
        self.logger.info('base name: ' + os.path.basename(apk))
        apk_name, apk_extension = os.path.splitext(apk)

        self.logger.info(apk_name)


        apk_name = os.path.basename(apk_name)

        if apk_name in examined:
            self.logger.error('Already examined ' + apk_name)
            return

        cmd = 'adb devices'
        os.system(cmd)
        self.logger.info(apk)

        # current_time = time.strftime(ISOTIMEFORMAT, time.localtime())
        par_dir = os.path.basename(os.path.abspath(os.path.join(apk, os.pardir)))  # the parent folder of the apk

        package = self.get_package_name(self.aapt_loc, apk)


        if not package:
            self.logger.error('Not a valid pkg.')
            return

        self.start_taintdroid(series)

        output_dir = self.out_base_dir + par_dir + '/' + apk_name + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filehandler = Utilities.set_file_log(self.logger, output_dir + 'UIExerciser_FlowIntent_FP_PY.log')
        self.logger.info('apk:' + apk)
        self.logger.info('pkg:' + package)

        self.run_adb_cmd('adb -s ' + series + ' shell "su 0 date -s `date +%Y%m%d.%H%M%S`"')

        UIExerciser.uninstall_pkg(package)
        UIExerciser.install_apk(apk)

        self.start_activities(package, csvpath, output_dir)

        self.uninstall_pkg(package)

        filehandler.close()
        self.logger.removeHandler(filehandler)

    def inspired_run(self, apk, examined):
        # apk = 'F:\\Apps\\COMMUNICATION\\com.mobanyware.apk'
        self.logger.info('base name: ' + os.path.basename(apk))
        apk_name, apk_extension = os.path.splitext(apk)

        self.logger.info(apk_name)
        if '_modified' not in apk_name:
            return
            # apk_modified = apk_name + '_modified.apk'
        else:
            apk_modified = apk
            apk_name = apk_name.replace('_modified', '')

        apk_name = os.path.basename(apk_name)

        if apk_name in examined:
            self.logger.error('Already examined ' + apk_name)
            return

        cmd = 'adb devices'
        os.system(cmd)
        self.logger.info(apk_modified)

        # current_time = time.strftime(ISOTIMEFORMAT, time.localtime())
        par_dir = os.path.basename(os.path.abspath(os.path.join(apk, os.pardir)))  # the parent folder of the apk

        package = self.get_package_name(self.aapt_loc, apk_modified)

        if not package:
            self.logger.error('Not a valid pkg.')
            return

        csvpath = self.get_csv_path(self.trigger_java_dir, par_dir, apk_name)
        if not os.path.isfile(csvpath):
            self.logger.error('tgt_Act.csv does not exist:' + csvpath)
            return

        output_dir = self.out_base_dir + par_dir + '/' + apk_name + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filehandler = Utilities.set_file_log(self.logger, output_dir + 'COSMOS_TRIGGER_PY.log')
        self.logger.info('apk:' + apk_modified)
        self.logger.info('pkg:' + package)
        self.logger.info('csv: ' + csvpath)

        UIExerciser.uninstall_pkg(package)
        UIExerciser.install_apk(apk_modified)

        self.start_activities(package, csvpath, output_dir)

        self.uninstall_pkg(package)

        filehandler.close()
        self.logger.removeHandler(filehandler)


