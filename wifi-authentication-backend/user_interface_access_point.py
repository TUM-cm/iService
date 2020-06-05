import time
import numpy
import random
import logging
import xmlrpclib
import threading
import crypto.nonce as nonce
from gui_access_point import GUI
from threading import Lock
from flask_wtf import FlaskForm
from flask_socketio import SocketIO
from flask import Flask, render_template, request
from hostapd_access_point import AccessPoint
from localvlc.testbed_setting import Testbed
from statistics_access_point import Statistics
from send_light.send_light_user import LightSender
from light_subscriber_access_point import LightSubscriber
from password_broadcast_access_point import PasswordBroadcast
from wtforms import StringField, SelectField, SubmitField

namespace = "/auth"
demo_ssid = "localvlc"

class Form(FlaskForm):
    led = SelectField(label="LED",
                      choices=[("Directional", "Directional"),
                               ("Pervasive", "Pervasive")])
    ssid = StringField("SSID", render_kw={'readonly': True})
    password_refresh_rate = SelectField(label="Password refresh rate",
                                        choices=[ ("5", "5 sec."), ("10", "10 sec."),
                                                 ("30", "30 sec."), ("60", "1 min."),
                                                 ("120", "2 min."), ("300", "5 min."),
                                                 ("600", "10 min.")])
    start = SubmitField("Start")
    stop = SubmitField("Stop")

app = Flask(__name__)
app.config['SECRET_KEY'] = nonce.gen_nonce_uuid4()
socketio = SocketIO(app)
run = True
thread_lock = Lock()

gui = None
thread = None
statistics = None
rpc_client = None
access_point = None
light_sender = None
light_subscriber = None
password_broadcast = None

def background_gui():
    while run:
        for field, value in gui.get_values().items():            
            if "img" in field:
                socketio.emit(field, {'y': value[0], 'x': value[1]}, namespace=namespace)
            else:
                socketio.emit(field, value, namespace=namespace)
        socketio.sleep(0.1)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = Form(request.form)
    form.ssid.data = demo_ssid
    disable_start = False
    if form.start.data:
        disable_start = True
        statistics.reset()
        logging.info("start action")
        led = Testbed.Directed_LED
        if "Pervasive" in form.led.data:
            led = Testbed.Pervasive_LED
        logging.info("start remote light receiver")
        rpc_client.start_light_receiver(led.get_sampling_interval())
        logging.info("configure light sender")
        light_sender.set_time_base_unit(led.get_time_base_unit())
        logging.info("start light subscriber for remote client")
        light_subscriber.start()
        logging.info("start access point")
        password_broadcast.start(form.ssid.data, form.password_refresh_rate.data)
        gui.set_status("Access point running ...")
    elif form.stop.data:
        logging.info("stop action")
        logging.info("stop remote light receiver")
        rpc_client.stop_light_receiver()
        logging.info("stop password broadcast")
        password_broadcast.stop()
        logging.info("stop light subscriber")
        light_subscriber.stop()
        logging.info("stop light sender")
        light_sender.stop()
        logging.info("stop access point")
        access_point.stop()
        gui.set_status("Access point stopped")
    return render_template('index.html', form=form,
                           disable_start=disable_start,
                           disable_stop=not disable_start)

@socketio.on('connect', namespace=namespace)
def on_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_gui)

def start(send_kernel_module, host_light_receiver):
    global gui
    global statistics
    global rpc_client
    global access_point
    global light_sender
    global light_subscriber
    global password_broadcast
    
    gui = GUI()
    statistics = Statistics(gui)
    access_point = AccessPoint(gui, statistics)
    light_sender = LightSender(send_kernel_module)
    light_subscriber = LightSubscriber(host_light_receiver, gui, statistics)
    password_broadcast = PasswordBroadcast(light_sender, gui, access_point, statistics)
    rpc_client = xmlrpclib.ServerProxy('http://' + host_light_receiver + ':8000')    
    socketio.run(app, host='0.0.0.0')

def test_plots(data_len=600):
    while True:
        voltage = numpy.random.randint(100, 1000, size=(1, data_len))[0]
        gui.set_light_signal(voltage)

def test_performance():
    while True:
        statistics.set_start_data_transmission()
        statistics.set_start_wifi_authentication()
        time.sleep(random.uniform(0.1, 6))
        statistics.update_data_transmission()
        statistics.update_wifi_authentication()

def test_start():
    global gui
    global statistics
    
    gui = GUI()
    statistics = Statistics(gui)
    gui.set_status("Access point running ...")
    gui.set_received_authentication("MobiSys", "123456")
    gui.set_generated_password("123456")
    gui.set_morse_data("morse data data data data ...")
    statistics.increase_authentication_success()
    statistics.increase_authentication_success()
    statistics.increase_authentication_fail()
    gui.add_client_action("wlan0", "01:02:03:04:05:06", "connect")
    gui.add_client_action("wlan0", "01:02:03:04:05:06", "disconnect")
    gui.add_client_action("wlan0", "01:02:03:04:05:06", "not authorized")
    threading.Thread(target=test_plots).start()
    threading.Thread(target=test_performance).start()
    socketio.run(app, host='0.0.0.0', debug=True)

if __name__ == "__main__":
    test_start()
