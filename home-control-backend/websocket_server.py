from websocket import create_connection
from flask import Flask, request

# https://stackoverflow.com/questions/44706033/connecting-to-a-flask-websocket-using-python

app = Flask(__name__)

@app.route('/2fa')
def twofa():
    client_ip = request.environ['REMOTE_ADDR']
    ws = create_connection("ws://" + client_ip + ":5000/light_token")
    #ws = create_connection("ws://" + client_ip + ":5000/light_token")
    data = []
    while len(data) < 10:
        data.append(ws.recv())
    ws.close()
    return ", ".join(data)
    
    # Old socketIO, websockets faster and easier to use
    #socketIO = socketIO_client.SocketIO(client_ip, 5000)
    #with socketIO_client.SocketIO(client_ip, 11234) as socketIO:
    #    socketIO.on('local_client', receive_key)
    #    socketIO.emit('home_control', {'data': 'request'})
    #    socketIO.wait_for_callbacks(seconds=1)
    #    socketIO.wait(seconds=0.1)
    
@app.route('/')
def index():
    return '<a href="/2fa">Two Factor Auth</a>'

demo = True
port = 8000
if demo:
    host='192.168.137.1'
else:
    host = 'localhost'

app.run(host=host, port=port)
