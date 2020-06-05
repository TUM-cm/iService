import os
import shutil

src_directory = "home_control/"
config_file = "config.py"
#hotspot_file = "hotspot.conf"
shutil.copy2(src_directory + config_file, os.getcwd())
#shutil.copy2(src_directory + hotspot_file, os.getcwd())

import sys
import logging
#import utils.log as log
#import localvlc.files as files
import home_control.app as home_control
import utils.kernel_module as kernel_module
#import utils.wifi_connector as wifi_connector
from utils.input import Input
from utils.config import Config
from send_light.broadcast_light_token import BroadcastLightToken

# python -m home_control.start_home_control config=config.ini

if not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

if len(sys.argv) != 2:
    sys.exit("\nWrong number of input parameters\n")

Input(sys.argv)
path = os.path.dirname(__file__)
config_general = Config(path, "General")
config_home_control_server = Config(path, "Home Control Server")
#log.activate_file_console(files.file_light_token_broadcast)
#log.activate_debug()

kernel_module_send_light = config_home_control_server.get("Send kernel module")
shutil.copy2("send_light/" + kernel_module_send_light, os.getcwd())

#if not os.path.isfile(hotspot_file):
#    sys.exit("\nHostapd file is not available\n")

#if not wifi_connector.interface_available():
#    sys.exit("\nWifi interface is not available\n")
kernel_module.add_light_sender()

# find hostapd process to kill: ps -ef | grep hostapd and then kill process id
#wifi_connector.activate_hotspot()

# evaluate=True for token distribution, evaluate=False for authorization test
broadcast_light_token = BroadcastLightToken(kernel_module_send_light,
                                            config_home_control_server.get("Sender time base unit"),
                                            token_period=config_home_control_server.get("Token period"),
                                            token_num=config_home_control_server.get("Token num"),
                                            evaluate=config_home_control_server.get("Evaluate Token Broadcast"))
broadcast_light_token.start()
logging.info("start broadcast light token")

home_control.set_data(config_general, config_home_control_server, broadcast_light_token)
logging.info("start home control")
home_control.start()
