import numpy
from flask import Flask
from threading import Lock
from flask import render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret!"
socketio = SocketIO(app)
thread = None
thread_lock = Lock()
namespace="/test"

def background_gui():
    data_len = 600
    while True:
        y = numpy.random.randint(100, 300, size=(1, data_len))[0]
        x = [""] * len(y)
        socketio.emit("img_light_signal", {'y': y.tolist(), 'x': x}, namespace=namespace)
        socketio.sleep(0.1)

@app.route("/")
def line_chart():
    return render_template('line_chart.html')

@socketio.on('connect', namespace=namespace) # '/test'
def on_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_gui)

if __name__ == "__main__":
    socketio.run(app, debug=True)
