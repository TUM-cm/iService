import os
import sys
import json
import pexpect
import server
import client
import logging
import netifaces
from flask import Flask
import utils.wifi_connector

logging.basicConfig(filename="relay-attack.log", level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Testbed with Windows laptop and smartphone
# server setup
# 1. start hotspot at Windows laptop (network: relay)
# 2. start server via control_testbed.py
# client setup
# 1. connect smartphone with relay network
# 2. start up for relay measurements

# Testbed with BeagleBone Black (deprecated)
# setup USB Wi-Fi adapter at server and client, see README.md (#WiFi)
# server setup
# 1. ssh debian@vlc03.cm.in.tum.de (temppwd)
# 2. python -m coupling.relay_attack.control true
# 3. result of latency measurements in local directory under testbed name
# 4. In case of blocked connection: killall wpa_supplicant; apt-get remove wpasupplicant
# client setup
# 1. ssh debian@vlc02.cm.in.tum.de (temppwd)
# 2. Debian Buster automatically recognizes WLAN adapter with correct name (w/o udev)
# 3. nohup python -m coupling.relay_attack.control false &
# 4. connect smartphone via USB to BBB
# 5. start Android application to enable USB tethering and allow latency measurements
# IMPORTANT: After boot of BBB, remove and attach Wi-Fi adapter

server_mode = (sys.argv[1].lower() == "true")

# Details about relay measurements
activate_tls = True
test_rounds = 50
port = 10000
test_length = 4000
hostname = None # localhost | None to infer local IP

# Details about WiFi interface
ssid = "relay"
ap_password = "helloworld"
ap_config_name = "hostapd.conf"
net = "10.0.0."
server_ip = net + "1"
wlan_interface = "wlan0"

if "linux" in sys.platform and not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

app = Flask(__name__)
@app.route('/start/<string:testbed>/<int:position>')
def start_sensing(testbed, position):
    try:
        testbed = "testbed_" + testbed.replace(" ", "_")
        logging.info("start sensing")
        logging.info(testbed)
        logging.info("position: " + str(position))
        if activate_tls:
            tls_hostname = "localhost"
            client.start_tls(hostname, tls_hostname, port, test_length, testbed, position)
        else:
            client.start_tcp(hostname, port, test_length, testbed, position)
        return json.dumps(True)
    except Exception as e:
        logging.debug("error in start sensing: " + str(e))
        return json.dumps(False)
    
# ifconfig wlan0 up 10.0.0.1 netmask 255.255.255.0
def main():
    global hostname
    if server_mode:
        if "linux" in sys.platform:
            logging.info("setup wlan interface")
            pexpect.run("ifconfig wlan0 up " + server_ip + " netmask 255.255.255.0") # fit to dhcp config
            utils.wifi_connector.activate_dhcp()
            logging.info("start access point")
            path_hostapd_template = os.path.join(__location__, "hostapd_template.conf")
            if not os.path.isfile(path_hostapd_template):
                sys.exit("\nHostapd template is not available\n")         
            if not utils.wifi_connector.interface_available():
                sys.exit("\nWifi interface is not available\n")
            config_template = open(path_hostapd_template, "rU").read()
            config = config_template.replace("<ssid>", ssid)
            config = config.replace("<password>", ap_password)
            path_hostapd = os.path.join(__location__, ap_config_name)
            logging.debug("hostapd config: " + path_hostapd)
            hostapd_config = open(path_hostapd, "w+")
            hostapd_config.write(config)
            hostapd_config.close()
            hostapd = pexpect.spawn("hostapd " + path_hostapd) # variable important to run process
        logging.info("start server")
        if activate_tls:
            server.start_tls(port, test_rounds)
        else:
            server.start_tcp(port, test_rounds)
    else: # client mode
        hostname = server_ip
        if not utils.wifi_connector.is_connected(ssid):
            logging.info("connect wifi")
            result = utils.wifi_connector.connect(ssid, ap_password)
            logging.info("wifi: {}".format(result))
        logging.info("Check IP of wlan interface")
        local_ip = "0"
        try:
            local_ip = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr']
        except:
            pass
        while not local_ip.startswith(net):
            try:
                local_ip = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr']
                logging.info("request DHCP")
                pexpect.run("dhclient " + wlan_interface)
            except:
                pass
        logging.info("start client")
        app.run(host="0.0.0.0")
    
if __name__ == "__main__":
    main()
    