from __future__ import division
import time
import logging

class Statistics(object):
    
    def __init__(self, gui):
        self.gui = gui
        self.initialize()
    
    def initialize(self):
        self.authentication_success = 0
        self.authentication_fail = 0
        self.duration_data_transmission = [0]
        self.duration_wifi_authentication = [0]
        self.start_data_transmission = None
        self.start_wifi_authentication = None
        self.current_password = None
    
    def reset(self):
        self.initialize()
        self.gui.reset()
    
    def increase_authentication_success(self):
        self.authentication_success += 1
        self.gui.set_success_fail_rate(self.ratio_authentication_success(),
                                       self.ratio_authentication_fail())
    
    def increase_authentication_fail(self):
        self.authentication_fail += 1
        self.gui.set_success_fail_rate(self.ratio_authentication_success(),
                                       self.ratio_authentication_fail())
    
    def ratio_authentication_success(self):
        return self.__ratio(self.authentication_success)
    
    def ratio_authentication_fail(self):
        return self.__ratio(self.authentication_fail)
    
    def __ratio(self, rate):
        return rate / (self.authentication_success + self.authentication_fail)
    
    def set_start_data_transmission(self):
        self.start_data_transmission = time.time()
        logging.debug("start: " + str(self.start_data_transmission))
    
    def set_start_wifi_authentication(self):
        self.start_wifi_authentication = time.time()
    
    def update_data_transmission(self, stop_time):
        if self.start_data_transmission != None:
            duration = stop_time - self.start_data_transmission
            self.start_data_transmission = None
            self.duration_data_transmission.append(duration)
            logging.debug("duration data transmission: " + ', '.join(map(str, self.get_data_transmission())))
            self.gui.set_data_transmission(self.get_data_transmission())
    
    def update_wifi_authentication(self):
        if self.start_wifi_authentication != None:
            duration = time.time() - self.start_wifi_authentication
            self.start_wifi_authentication = None
            self.duration_wifi_authentication.append(duration)
            logging.debug("duration wifi authentication: " + ', '.join(map(str, self.get_wifi_authentication())))
            self.gui.set_wifi_authentication(self.get_wifi_authentication())
    
    def get_wifi_authentication(self):
        return self.duration_wifi_authentication
    
    def get_data_transmission(self):
        return self.duration_data_transmission

    def set_current_password(self, password):
        self.current_password = password
    
    def get_current_password(self):
        return self.current_password
