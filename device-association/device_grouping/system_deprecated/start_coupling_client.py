from localvlc.testbed_setting import Testbed
import utils.wifi_connector as wifi_connector
from service_discovery import MessageServiceDiscovery
from receive_light.light_control import ReceiveLightControl
import coupling.device_grouping.online.static.coupling_client as coupling_client

server_port = 1026
service_discovery = MessageServiceDiscovery()

def parse_service_discovery(morse_msg):
    morse_msg = morse_msg
    #print("msg: <{}>".format(morse_msg))
    try:
        service_discovery.parse(morse_msg)
        ssid = service_discovery.get_location()
        pwd = service_discovery.get_password()
        print("ssid: {}".format(ssid)) # ssid
        print("password: {}".format(pwd))
        wifi = wifi_connector.connect(ssid, pwd)
        print("wifi: {}".format(wifi))
        if wifi:
            server_ip = wifi_connector.get_ip("wlan0")
            server_ip = server_ip.split(".")
            server_ip[-1] = str(1) # gateway
            server_ip = ".".join(server_ip)
            coupling_client.start(server_ip, server_port)
    except TypeError:
        pass

led = Testbed.Pervasive_LED
light_control = ReceiveLightControl(
    ReceiveLightControl.Action.morse_callback, sampling_interval=led.get_sampling_interval(), callback_morse=parse_service_discovery)
light_control.start()
