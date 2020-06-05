from flask import Flask
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# https://github.com/kennethreitz/flask-sockets

app = Flask(__name__)
sockets = Sockets(app)
counter = 0

#@sio.on('home_control')
#def request_key(data):
#    print('message ', data)
#    emit('local_client', {'key': 42})

@sockets.route('/')
@sockets.route('/light_token')
def send_light_token(ws):
    global counter
    while not ws.closed:
        print "light token"
        ws.send("light token: " + str(counter))
        counter += 1
        
@app.route('/')
def index():
    return 'hello world'

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('localhost', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
