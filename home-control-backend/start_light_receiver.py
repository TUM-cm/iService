import os
import sys
import shutil
import logging
import utils.log as log
import localvlc.files as files
import utils.wifi_connector as wifi_connector
import utils.kernel_module as kernel_module
from utils.input import Input
from utils.config import Config
from light_receiver_gateway import LightReceiverGateway
from receive_light.light_control import ReceiveLightControl

# python -m home_control.start_light_receiver config=config.ini

if not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

if len(sys.argv) != 2:
    sys.exit("\nWrong number of input parameters\n")

if not wifi_connector.interface_available():
    sys.exit("\nWifi interface is not available\n")

logging.info("activate wifi interface")
wifi_connector.activate_wifi()

Input(sys.argv)
path = os.path.dirname(__file__)
config_general = Config(path, "General")
config_home_control_client = Config(path, "Home Control Client")

kernel_module_receive_light = config_home_control_client.get("Receive kernel module")
shutil.copy2("receive_light/" + kernel_module_receive_light, os.getcwd())
kernel_module.check(kernel_module_receive_light)

log.activate_file_console(files.file_light_token_receiver)

target_ap = config_home_control_client.get("Target AP")
password_ap = config_home_control_client.get("Password AP")

if wifi_connector.connect(target_ap, password_ap):
    # Start to continuously record light tokens in background
    light_control = ReceiveLightControl(ReceiveLightControl.Action.select_token,
                                        sampling_interval = config_home_control_client.get("Sampling interval"),
                                        evaluate_home_control=True)
    light_control.start()
    logging.info("light control started ...")
    
    light_gateway = LightReceiverGateway(light_control, config_general)
    logging.info("light gateway started ...")
    light_gateway.start()
