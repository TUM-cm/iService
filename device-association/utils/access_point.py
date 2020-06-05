import re
import os
import io
import time
import signal
import pexpect
import logging
import threading
from utils.two_way_dict import TwoWayDict

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class AccessPoint:
    
    MAC = "[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}"
    IP = "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    
    PATTERN_DHCP = "^(" + MAC + ") +(" + IP + ").*$"
    NUM_DHCP_ELEMENTS = 2
    
    PATTERN_HOSTAPD = "^(wlan\d): (.*) (" + MAC + ")$"
    NUM_HOSTAPD_ELEMENTS = 3
    
    def __init__(self, config_name="hostapd.conf"):
        self.hostapd_child = None
        self.clients = TwoWayDict()     
        self.config_name = config_name
        # https://stackoverflow.com/questions/13954840/how-do-i-convert-lf-to-crlf
        self.config_template = open(os.path.join(__location__, "hostapd_template.conf"), "rU").read()
        self.deny_hosts = io.open(os.path.join(__location__, "hostapd.deny"), "a", newline="\n")
    
    def reset_deny_hosts(self):
        self.deny_hosts.seek(0)
        self.deny_hosts.truncate()
    
    def deny_hosts(self, macs, sleep_period=0.2):
        self.reset_deny_hosts()
        for mac in macs:
            self.deny_hosts.write(mac + "\n")
        time.sleep(sleep_period)
        for mac in macs:
            pexpect.run("hostapd_cli disassociate " + mac)
        #prerequisite: define ctrl interface in hostapd.conf
        #ctrl_interface=/var/run/hostapd
        #hostapd_cli deauthenticate 00:5e:3d:38:fe:ab
        #hostapd_cli disassociate 01:23:45:67:89:AB
    
    def stop(self):
        logging.info("stop access point")
        self.run = False        
        self.hostapd_child.kill(signal.SIGTERM)
    
    def __start(self, ssid, password):
        logging.info("start access point ...")
        config = self.config_template.replace("<ssid>", ssid)
        config = config.replace("<password>", password)
        hostapd_config = open(os.path.join(__location__, self.config_name), "w")
        hostapd_config.write(config)
        hostapd_config.close()
        self.run = True
        self.hostapd_child = pexpect.spawn("hostapd " + os.path.join(__location__, self.config_name))
        while self.run:
            self.hostapd_child.expect("\n", timeout=None)
            output = self.hostapd_child.before.strip()
            match_action = re.findall(AccessPoint.PATTERN_HOSTAPD, output)
            "wlan0: AP-STA-CONNECTED ac:37:43:4f:f7:35"
            "wlan0: AP-STA-DISCONNECTED ac:37:43:4f:f7:35"
            if match_action and len(match_action[0]) == AccessPoint.NUM_HOSTAPD_ELEMENTS:
                parts  = match_action[0]
                action = parts[1]
                mac = parts[2]
                if "CONNECTED" in action:
                    self.find_connected_clients()
                elif "DISCONNECTED" in action:
                    del self.clients[mac]
    
    # hostapd_cli all_sta     client MAC addresses
    # arp -a -n               client IP addresses
    
    # MAC                IP              hostname       valid until         manufacturer
    # ===============================================================================================
    # ac:37:43:4f:f7:35  10.10.0.2        -NA-           2018-07-11 10:05:30 -NA-
    def find_connected_clients(self):
        self.clients = []
        out = pexpect.run("dhcp-lease-list --lease /var/lib/dhcp/dhcpd.leases")
        for line in out.split("\n"):
            match = re.findall(AccessPoint.PATTERN_DHCP, line)
            if match and len(match[0]) == AccessPoint.NUM_DHCP_ELEMENTS:
                mac = match[0][0]
                ip = match[0][1]
                self.clients[ip] = mac
    
    def get_num_connected_clients(self):
        return len(self.clients)
    
    def get_mac(self, ip):
        return self.clients[ip]
    
    def start(self, ssid, password):
        #self.__start(ssid, password)
        thread = threading.Thread(target=self.__start, args=[ssid, password])
        thread.start()

def main():
    logging.basicConfig(level=logging.DEBUG)
    AccessPoint().start("ssid", "12345693")
    
if __name__ == "__main__":
    main()
