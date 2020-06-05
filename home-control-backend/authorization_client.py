import json
import logging
import threading
from flask import Flask
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# https://github.com/kennethreitz/flask-sockets

class AuthorizationClient():
    
    def __init__(self, light_receiver):
        self.app = Flask(__name__)
        self.sockets = Sockets(self.app)
        self.light_receiver = light_receiver
        self.sockets.add_url_rule("/authorization", "authorization", self.authorization)
    
    def authorization(self, ws):
        logging.info("authorization request")
        challenge = ws.receive()
        logging.info("challenge: " + challenge)
        if "request_authorization" in challenge:
            nonce = int(challenge.split(":")[1])
            nonce += 1
            logging.info("inc. nonce: " + str(nonce))
            json_encrypted_nonce = self.light_receiver.get_encrypted_data(nonce)
            encrypted_nonce = json.loads(json_encrypted_nonce.data)["result"]
            logging.info("encrypted nonce: " + encrypted_nonce)
            ws.send(encrypted_nonce)
    
    def start_server(self, address=('0.0.0.0', 5000)):
        logging.info("start authorization client: " + str(address))
        server = pywsgi.WSGIServer(address, self.app, handler_class=WebSocketHandler)
        server.serve_forever()
    
    def start(self):
        thread = threading.Thread(target=self.start_server)
        thread.start()

if __name__ == "__main__":
    authorization_client = AuthorizationClient(None)
    authorization_client.start()
