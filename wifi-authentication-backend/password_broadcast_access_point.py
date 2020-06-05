import time
import sched
import logging
import threading
import utils.log as log
from crypto.otp import OneTimePass

class PasswordBroadcast(object):
    
    def __init__(self, light_sender, gui, access_point, statistics, prio=1):
        self.light_sender = light_sender
        self.gui = gui
        self.access_point = access_point
        self.statistics = statistics
        self.priority = prio
        self.broadcast_event = None 
        self.scheduler = sched.scheduler(time.time, time.sleep)
    
    def stop(self):
        logging.info("stop password broadcast")
        if not self.scheduler.empty() and self.broadcast_event:
            self.scheduler.cancel(self.broadcast_event)
    
    def __start(self, restart):
        logging.info("start password broadcast ...")
        if restart:
            self.light_sender.stop()
            self.access_point.stop()
            self.gui.set_generated_password("")
        password = self.totp.get_token()
        data = self.ssid + password
        logging.debug("broadcast: " + data)
        self.statistics.set_current_password(password)
        self.light_sender.set_data(data)
        self.statistics.set_start_data_transmission()
        self.light_sender.start()
        self.statistics.set_start_wifi_authentication()
        self.access_point.start(self.ssid, password)
        self.gui.set_generated_password(password)
        self.broadcast_event = self.scheduler.enter(self.password_refresh_period,
                                                    self.priority,
                                                    self.__start, (True,))
        self.scheduler.run()
    
    def start(self, ssid, password_refresh_period, token_len=8):
        self.ssid = ssid
        self.password_refresh_period = int(password_refresh_period)
        self.totp = OneTimePass(interval=self.password_refresh_period, token_len=token_len)
        if self.light_sender.check_active():
            self.light_sender.stop()
        thread = threading.Thread(target=self.__start, args=(False,))
        thread.start()

def main():
    log.activate_debug()
    from demo_wifi_login.gui_access_point import GUI
    from send_light.send_light_user import LightSender
    from demo_wifi_login.hostapd_access_point import AccessPoint
    from demo_wifi_login.statistics_access_point import Statistics
    gui = GUI()
    light_sender = LightSender()
    statistics = Statistics(gui)
    access_point = AccessPoint(gui, statistics)
    password_broadcast = PasswordBroadcast(light_sender, gui, access_point, statistics)
    password_broadcast.start("LocalVLC", 120)

if __name__ == "__main__":
    main()
