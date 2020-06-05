import time
import numpy
import logging
import threading
import paho.mqtt.client as mqtt

class LightSubscriber:
    
    def __init__(self, host, gui, statistics,
                 topic_light_signal="light_signal",
                 topic_authentication="wifi_authentication",
                 topic_morse="morse_data"): # topic_time_data="time_data_transmission"
        self.host = host
        self.gui = gui
        self.statistics = statistics
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.message_callback_add(topic_light_signal, self.on_message_light_signal)
        self.mqtt_client.message_callback_add(topic_authentication, self.on_message_authentication)
        self.mqtt_client.message_callback_add(topic_morse, self.on_message_morse)
        #self.mqtt_client.message_callback_add(topic_time_data, self.on_message_time_data)
    
    def on_connect(self, mqttc, userdata, flags, rc):
        logging.debug("light subscriber connect")
        self.mqtt_client.subscribe("#")
    
    def on_message_time_data(self, mosq, obj, msg):
        logging.debug("message time data")
        stop_time = float(msg.payload)
        logging.debug("stop:" + stop_time)
        self.statistics.update_data_transmission(stop_time)
    
    def on_message_light_signal(self, mosq, obj, msg):
        logging.debug("message light signal")        
        voltage = numpy.fromstring(msg.payload, dtype=numpy.int)
        self.gui.set_light_signal(voltage)
    
    def on_message_authentication(self, mosq, obj, msg):
        logging.debug("message wifi authentication")
        ssid, password = msg.payload.split(":")
        if self.statistics.get_current_password() == password:
            self.statistics.update_data_transmission(time.time())
        self.gui.set_received_authentication(ssid, password)
    
    def on_message_morse(self, mosq, obj, msg):
        logging.debug("message Morse data")
        self.gui.set_morse_data(msg.payload)
    
    def __start(self):
        self.mqtt_client.connect(self.host)
        self.mqtt_client.loop_forever()
    
    def start(self):
        logging.info("start light subscriber ...")
        thread = threading.Thread(target=self.__start)
        thread.start()
    
    def stop(self):
        logging.info("stop light subscriber")
        self.mqtt_client.disconnect()
