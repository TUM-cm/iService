import numpy
import datetime
from utils.custom_enum import enum_name

field_names = ["status", "generated_password", "actions",
               "authentication_success", "authentication_fail",
               "duration_data", "img_duration_data",
               "duration_wifi", "img_duration_wifi",
               "received_ssid", "received_password",
               "morse_data", "img_light_signal"]

fields = enum_name(*field_names)

class GUI:
    
    def __init__(self):
        self.fields = dict()
        self.num_action_columns = 4        
        self.row_format ="{:25}" * self.num_action_columns
        self.initialize()
    
    def initialize(self):
        for name in field_names:
            if "img" in name:
                self.fields[name] = ([0], [0])
            else:
                self.fields[name] = ""
        self.__set_value(fields.actions,
                         "<b>" + self.row_format.format("Time", "Interface", "MAC", "Action") + "</b>")
    
    def reset(self):
        self.initialize()        
    
    def __set_value(self, key, value):
        self.fields[key] = value
    
    def get_values(self):
        return self.fields
    
    def set_status(self, status):
        self.__set_value(fields.status, status)
    
    def set_received_authentication(self, ssid, password):
        self.__set_value(fields.received_ssid, ssid)
        self.__set_value(fields.received_password, password)
    
    def set_generated_password(self, password):
        self.__set_value(fields.generated_password, password)
    
    def set_morse_data(self, data):
        self.__set_value(fields.morse_data, data)
    
    def set_success_fail_rate(self, ratio_success, ratio_fail, digits=2):
        self.__set_value(fields.authentication_success, round(ratio_success, digits))
        self.__set_value(fields.authentication_fail, round(ratio_fail, digits))
    
    def set_light_signal(self, voltage):
        time = [""] * voltage.shape[0]
        self.__set_value(fields.img_light_signal, (voltage.tolist(), time))
    
    def add_client_action(self, interface, mac, action):
        old = self.fields[fields.actions] + "<br>"
        new = self.row_format.format(datetime.datetime.now().strftime("%H:%M:%S.%f"),
                                     interface, mac, action)
        self.__set_value(fields.actions, old + new)
    
    def set_wifi_authentication(self, wifi_authentication):
        self.__set_duration_performance(fields.duration_wifi,
                                        fields.img_duration_wifi,
                                        wifi_authentication)
    
    def set_data_transmission(self, data_transmission):        
        self.__set_duration_performance(fields.duration_data,
                                        fields.img_duration_data,
                                        data_transmission)
    
    def __set_duration_performance(self, field_value, field_img, values, digits=2, unit=" s"):
        xlabels = list(range(1, len(values)+1))
        self.__set_value(field_value,
                       str(round(numpy.mean(values), digits)) + " +/ "
                       + str(round(numpy.std(values), digits)) + unit)
        self.__set_value(field_img, (values, xlabels))
    