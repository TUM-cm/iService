import numpy
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import coupling.light_grouping_pattern.light_analysis as light_analysis
from utils.plot import RealtimePlot

# Run MQTT publisher at IoT board receiving the raw light signal
# Plot light signal at end device (no X11 at IoT board, no plotting)

class StreamConnector:
    
    def __init__(self, ip, on_message, subscribe_topic="light_signal"):
        self.ip = ip
        self.mqtt_client = mqtt.Client()
        self.subscribe_topic = subscribe_topic
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = on_message
    
    def on_connect(self, client, userdata, flags, rc):
        print("connect: " + client._host)
        self.mqtt_client.subscribe(self.subscribe_topic)
    
    def start(self):
        self.mqtt_client.connect(self.ip)
        self.mqtt_client.loop_forever()
    
    def stop(self):
        self.mqtt_client.disconnect()

class Plot:
    
    def __init__(self, ip):
        self.connector = StreamConnector(ip, self.on_message_plot)
        self.realtime_plot = RealtimePlot(self.__handle_close_fig)

    def __handle_close_fig(self, _):
        self.mqtt_client.disconnect()
    
    def on_message_plot(self, client, userdata, msg):
        data = numpy.fromstring(msg.payload)
        data_len = data.shape[0] // 2
        array_reshape = data.reshape(2, data_len)
        voltage = array_reshape[0].astype(numpy.int)
        #voltage_time = array_reshape[1].astype(numpy.uint)
        self.realtime_plot.plot(voltage)
    
    def start(self):        
        self.connector.start()

class DataAnalysis:
    
    def __init__(self, ip, plot=False, plot_data_len=2000):
        self.plot = plot
        self.plot_data_len = plot_data_len
        self.connector = StreamConnector(ip, self.on_message_analysis)
    
    def start(self):
        self.connector.start()
    
    def on_message_analysis(self, client, userdata, msg):
        self.connector.stop()
        data = numpy.fromstring(msg.payload)       
        data_len = data.shape[0] // 2
        array_reshape = data.reshape(2, data_len)
        voltage = array_reshape[0].astype(numpy.int)
        voltage_time = array_reshape[1].astype(numpy.uint)
        if self.plot:
            plt.plot(voltage_time[:self.plot_data_len], voltage[:self.plot_data_len])
            plt.show()
        print(light_analysis.get_sequence(voltage, voltage_time))
        light_analysis.detect_cycle_by_min_correlation(voltage)
        light_analysis.detect_cycle_by_max_min_correlation(voltage)

# public MQTT broker: test.mosquitto.org, broker.hivemq.com
def test_data_analysis():
    ip_light_receiver = "192.168.2.39"
    data_analysis = DataAnalysis(ip_light_receiver)
    data_analysis.start()

def test_realtime_plot():
    ip_light_receiver = "131.159.24.92"
    plot = Plot(ip_light_receiver)
    plot.start()

def main():
    test_realtime_plot()
    
if __name__ == "__main__":
    main()
