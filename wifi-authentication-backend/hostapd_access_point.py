import re
import os
import signal
import pexpect
import logging
import threading

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class AccessPoint(object):
    
    MAC = "[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}:[a-f0-9]{2}"
    PATTERN_CONNECT_DISCONNECT = "^(wlan\d): (.*) (" + MAC + ")$"
    PATTERN_FAILED_CONNECT = "^(wlan\d): STA (" + MAC + ") .*: (deauthenticated due to local deauth request)$"
    
    def __init__(self, gui, statistics):
        self.gui = gui
        self.hostapd_child = None
        self.statistics = statistics
        # https://stackoverflow.com/questions/13954840/how-do-i-convert-lf-to-crlf
        self.config_name = "hostapd.conf"
        self.config_template = open(os.path.join(__location__, "hostapd_template.conf"), "rU").read()
    
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
            #logging.debug("<" + output + ">")
            "wlan0: STA ac:37:43:4f:f7:35 IEEE 802.11: deauthenticated due to local deauth request"
            "wlan0: AP-STA-CONNECTED ac:37:43:4f:f7:35"
            "wlan0: AP-STA-DISCONNECTED ac:37:43:4f:f7:35"
            match_action = re.findall(AccessPoint.PATTERN_CONNECT_DISCONNECT, output)
            match_fail = re.findall(AccessPoint.PATTERN_FAILED_CONNECT, output)
            if match_action and len(match_action[0]) == 3:
                parts  = match_action[0]
                interface = parts[0]
                action = parts[1]
                mac = parts[2]
                self.performAction(interface, mac, action)
            elif match_fail and len(match_fail[0]) == 3:
                parts = match_fail[0]                
                interface = parts[0]
                mac = parts[1]
                action = parts[2]
                self.performAction(interface, mac, action)
    
    def performAction(self, interface, mac, action):
        logging.debug("-------- start ---------------")
        logging.debug("Access point")
        logging.debug("interface: " + interface)
        logging.debug("mac: " + mac)
        logging.debug("action: " + action)
        logging.debug("-------- end ---------------")
        self.gui.add_client_action(interface, mac, action)
        if "CONNECTED" in action:
            self.statistics.update_wifi_authentication()
            self.statistics.increase_authentication_success()
        elif "deauth" in action:
            self.statistics.increase_authentication_fail()
        
    def start(self, ssid, password):
        #self.__start(ssid, password)
        thread = threading.Thread(target=self.__start, args=[ssid, password])
        thread.start()

def main():
    from demo_wifi_login.gui_access_point import GUI
    from demo_wifi_login.statistics_access_point import Statistics
    gui = GUI()
    statistics = Statistics(gui)
    logging.basicConfig(level=logging.DEBUG)
    AccessPoint(gui, statistics).start("LocalVLC", "12345693")
    
if __name__ == "__main__":
    main()
