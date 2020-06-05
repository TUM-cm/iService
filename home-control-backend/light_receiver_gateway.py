import logging
from flask import Flask, jsonify
from crypto.speck_wrapper import Speck

class LightReceiverGateway(object):
    
    def __init__(self, light_control, config_general):
        self.light_control = light_control
        self.app = Flask(__name__)
        self.config_general = config_general
        self.crypto_block_size = int(self.config_general.get("Crypto block size"))
    
    # https://localhost:11234/api/light/token
    def get_token(self):
        return jsonify(result=self.light_control.get_token())
    
    # https://localhost:11234/api/light/encryption/1234
    def get_encrypted_data(self, data):
        key = self.light_control.get_token()
        logging.info("token: " + str(key))
        speck = Speck(key, self.crypto_block_size)
        cipherdata = speck.encrypt(data)
        return jsonify(result=cipherdata)
    
    def register_endpoints(self):
        self.app.add_url_rule("/api/light/encryption/<string:data>",
                              "get_encrypted_data", self.get_encrypted_data)
        self.app.add_url_rule("/api/light/token", "get_token", self.get_token)
        
    def start(self):
        self.register_endpoints()
        if self.config_general.get("TLS") == "True":
            self.app.run(host=self.config_general.get("Address"),
                         port=int(self.config_general.get("Port")),
                         ssl_context=(self.config_general.get("Certificate"),
                                      self.config_general.get("Key")))
        else:
            self.app.run(host=self.config_general.get("Address"),
                         port=int(self.config_general.get("Port")))
