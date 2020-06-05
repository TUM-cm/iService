import os
import sys
import pexpect
import logging
import utils.log as log
from crypto.otp import OneTimePass
from service_discovery import Service
import utils.wifi_connector as wifi_connector
from coupling.utils.access_point import AccessPoint
from service_discovery import MessageServiceDiscovery

# parameter
pwd_len = 8
wifi_ip = "10.0.0.1"
coupling_server_port = coupling_simulator.server_port

location = "0105038"
services = [Service("a", "abc"),
            Service("b", "def"),
            Service("c", "123")]

if not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

#log.activate_debug()
log.activate_info()
logging.info("activate network adapter")
pexpect.run("ifconfig wlan0 up " + wifi_ip + " netmask 255.255.255.0") # fit to dhcp config
wifi_connector.activate_dhcp()

if not os.path.isfile("demo_wifi_login/hostapd_template.conf"):
    sys.exit("\nHostapd template is not available\n")

if not wifi_connector.interface_available():
    sys.exit("\nWifi interface is not available\n")

totp = OneTimePass(token_len=pwd_len)
password = totp.get_token()
ssid = location

msg_service_discovery = MessageServiceDiscovery()
msg_service_discovery.construct(location, password, services)
msg_service_discovery.broadcast()

access_point = AccessPoint()
access_point.start(ssid, password)

coupling_server.start(coupling_server_port)
