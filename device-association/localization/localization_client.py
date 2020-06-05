# gateway: ip r | grep usb
import os
import requests
from requests.auth import HTTPBasicAuth

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Client:
    
    def __init__(self, server_ip, port, protocol):
        self.port = port
        self.server_ip = server_ip
        self.protocol = protocol
        self.host = self.server_ip + ":" + str(port)
    
    def get_url(self, api):
        return self.protocol + self.host + api

class LocalizationBoardClient(Client):
    
    def __init__(self, port, protocol="http://", usb_ip="192.168.42.129"):
        super().__init__(self, usb_ip, port, protocol)
    
    def get_measurement(self, ble=True, wifi=True, duration=10, api="/measurement"):
        url = self.get_url(api)
        payload = {"BLE": ble, "WIFI": wifi, "duration": duration}
        headers = {'content-type': 'application/json'}
        request = requests.post(url, json=payload, headers=headers)
        return request.content
    
class LocalizationServerClient(Client):
    
    def __init__(self, server_ip, port, protocol="http://"):
        super().__init__(self, server_ip, port, protocol)
    
    def within_room(self, data, api=""):
        url = self.get_url(api)
        request = requests.post(url, json=data, auth=HTTPBasicAuth('user', 'pass'))    
        return bool(request.content)

def main():
    import utils.wifi_connector
    from utils.serializer import JsonSerializer

    board_port = 8002
    localization_client = LocalizationBoardClient(board_port)
    
    server_port = 2000
    gateway_ip = "20"
    server_ip = utils.wifi_connector.get_ip("wlan0")
    server_ip = server_ip[:server_ip.rfind(".") + 1] + gateway_ip
    localization_server = LocalizationServerClient(server_ip, server_port)
    
    data = localization_client.get_measurement()
    path = os.path.join(__location__, "data")
    serializer = JsonSerializer(path)
    serializer.serialize(data)
    localization_server.within_room(data)

if __name__ == "__main__":
    main()
