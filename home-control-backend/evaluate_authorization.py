import os
import sys
import time
import json
import shutil
import logging
import requests
#import utils.log as log
import utils.kernel_module as kernel_module
from utils.input import Input
from utils.config import Config
from utils.serializer import DillSerializer
from localvlc.testbed_setting import Testbed
from localvlc.home_control.authorization_result import AuthorizationResult
from receive_light.light_control import ReceiveLightControl
from home_control.authorization_client import AuthorizationClient
from home_control.light_receiver_gateway import LightReceiverGateway

# python -m home_control.evaluate_authorization config=config.ini

requests.packages.urllib3.disable_warnings()

if not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

if len(sys.argv) != 2:
    sys.exit("\nWrong number of input parameters\n")

Input(sys.argv)
path = os.path.dirname(__file__)
config_general = Config(path, "General")
config_home_control_server = Config(path, "Home Control Server")
config_home_control_client = Config(path, "Home Control Client")

#log.activate_debug()

kernel_module_receive_light = config_home_control_client.get("Receive kernel module")
shutil.copy2("receive_light/" + kernel_module_receive_light, os.getcwd())
kernel_module.check(kernel_module_receive_light)

if config_general.get("TLS"):
    protocol = "https://"
else:
    protocol = "http://"
home_control_url = protocol + config_home_control_server.get("IP") + ":" + \
                config_general.get("Port") + "/"
home_control_authorization_url = home_control_url + "api/evaluate/authorization"
logging.info("Authorization url: " + home_control_authorization_url)

testbed_settings = {130000: Testbed.Directed_LED, 50000: Testbed.Pervasive_LED}
sender_time_base_unit = int(config_home_control_server.get("Sender time base unit"))
testbed_setting = testbed_settings[sender_time_base_unit]

light_control = ReceiveLightControl(ReceiveLightControl.Action.select_token,
                                    sampling_interval=testbed_setting.get_sampling_interval(),
                                    evaluate_home_control=True)
light_control.start()
logging.info("start receive light control")

light_receiver_gateway = LightReceiverGateway(light_control, config_general)
authorization_client = AuthorizationClient(light_receiver_gateway)
authorization_client.start()

config_home_control_client = Config(path, "Home Control Client")
authorization_rounds = int(config_home_control_client.get("Authorization rounds"))
authorization_results = list()

time.sleep(5) # to ensure that light receiver is up and running

logging.info("start evaluation")
for i in range(authorization_rounds):
    logging.info("round: " + str(i+1))
    start = time.time()
    r = requests.get(home_control_authorization_url,
                     verify=config_general.get("CA"))
    result = json.loads(r.text)["result"]
    stop = time.time()
    duration = (stop - start)
    logging.info("result: " + str(result) + ", duration: " + str(duration))
    authorization_results.append(AuthorizationResult(start, stop, duration, result))

basedir_results = "./evaluation/results/home_control/"
filename = "authorization_token_period_%s"
filename = filename % (config_home_control_server.get("Token period"))
serializer = DillSerializer(basedir_results + filename)
serializer.serialize(authorization_results)
logging.info("evaluation saved")
