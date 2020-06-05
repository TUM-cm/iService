import os
import re
import sys
#import time
import logging
import netifaces
import threading
#import subprocess
import utils.log as log
import paho.mqtt.client as mqtt
import utils.kernel_module as kernel_module
import receive_light.decoding.morse_code as morse_code
from utils.vlc_interfaces import ReceiverDevice
from SimpleXMLRPCServer import SimpleXMLRPCServer

class LightClient:
    
    def __init__(self, local_ip, logging=False):
        #self.stream_data = None
        self.last_password = None
        self.logging = logging
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(local_ip)
        self.rpc_server = SimpleXMLRPCServer((local_ip, 8000))
        #self.rpc_server.register_introspection_functions()
        self.rpc_server.register_function(self.start_light_receiver, 'start_light_receiver')
        self.rpc_server.register_function(self.stop_light_receiver, 'stop_light_receiver') 
    
    def start_light_receiver(self, sampling_interval):
        logging.debug("start light receiver")
        self.mqtt_client.loop_start()
        logging.debug("sampling interval: " + str(sampling_interval))
        
        # set page size for sending voltage and time
        #page_size = light_receiver.calc_page_size(sampling_interval)
        #logging.debug("page size: " + str(page_size))
        #self.stream_data = numpy.empty(shape=(2, page_size))
        
        #local_connector = LocalConnector(self.callback_light_signal, sampling_interval)
        #thread = threading.Thread(target=local_connector.start)
        #thread.start()
        #light_receiver.start_stream(ReceiverDevice.pd, sampling_interval, callback_light_signal)
        thread = threading.Thread(target=light_receiver.start_stream,
                                  args=(ReceiverDevice.pd, sampling_interval,
                                        self.callback_light_signal,
                                        self.logging,))
        thread.start()
        return True
    
    def stop_light_receiver(self):
        logging.debug("stop light receiver")
        self.mqtt_client.loop_stop()
        light_receiver.stop_stream()
        return True
    
    def callback_light_signal(self, voltage, voltage_time, data_len=600, len_password=8):
        logging.debug("callback light signal")        
        msg = morse_code.parse(voltage, voltage_time)
        parts = msg.split("\n")
        most_frequent_item = max(set(parts), key=parts.count)
        alpha_numeric = re.findall(r"[^\W\d_]+|\d+", most_frequent_item)
        if len(alpha_numeric) == 2 and len(alpha_numeric[1]) == len_password:
            ssid = alpha_numeric[0]
            password = alpha_numeric[1]
            # Time synchronization not precise enough below millisecond
            #if self.last_password != password:
            #    logging.debug("publish time data transmission")
            #    self.mqtt_client.publish("time_data_transmission", time.time())
            #    self.last_password = password
            logging.debug("publish wifi login")
            self.mqtt_client.publish("wifi_authentication", ssid + ":" + password)
        logging.debug("publish light signal")
        voltage_subset = voltage[:data_len]
        self.mqtt_client.publish("light_signal", voltage_subset.tostring())
        logging.debug("publish Morse data")
        self.mqtt_client.publish("morse_data", msg)
        #threading.Thread(target=self.publish_light_signal, args=[voltage, time]).start()
        #threading.Thread(target=self.publish_wifi_authentication, args=[voltage, time]).start()
    
    def publish_light_signal(self, voltage, voltage_time, data_len=600):
        logging.debug("publish light signal")
        #self.stream_data[0] = voltage
        #self.stream_data[1] = time
        #print morse_code.parse(voltage, time)
        #self.mqtt_client.publish("light_signal", self.stream_data.tostring())
        voltage_subset = voltage[:data_len]
        self.mqtt_client.publish("light_signal", voltage_subset.tostring())
    
    def publish_wifi_authentication(self, voltage, voltage_time, len_password=8):
        msg = morse_code.parse(voltage, voltage_time)
        logging.debug("publish Morse data")
        self.mqtt_client.publish("morse_data", msg)
        parts = msg.split("\n")
        most_frequent_item = max(set(parts), key=parts.count)
        alpha_numeric = re.findall(r"[^\W\d_]+|\d+", most_frequent_item)
        if len(alpha_numeric) == 2 and len(alpha_numeric[1]) == len_password:
            ssid = alpha_numeric[0]
            password = alpha_numeric[1]
            logging.debug("publish wifi login")
            self.mqtt_client.publish("wifi_authentication", ssid + ":" + password)
    
    def start(self):
        self.rpc_server.serve_forever()
    
if not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

#log.activate_debug()
log.activate_info()

kernel_module.add_light_receiver()

sys.path.append("./receive_light")
import light_receiver

local_ip = netifaces.ifaddresses("eth0")[netifaces.AF_INET][0]['addr']
#local_ip = netifaces.ifaddresses("usb0")[netifaces.AF_INET][0]['addr']
logging.debug(local_ip)
light_client = LightClient(local_ip)
logging.info("light receiver is running ...")
light_client.start()
