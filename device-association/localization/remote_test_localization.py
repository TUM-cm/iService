import utils.wifi_connector
from utils.serializer import JsonSerializer
from localization_client import LocalizationBoardClient
from localization_client import LocalizationServerClient

board_port = 8002
localization_client = LocalizationBoardClient(board_port)

server_port = 2000
gateway_ip = "20"
server_ip = utils.wifi_connector.get_ip("wlan0")
server_ip = server_ip[:server_ip.rfind(".")+1] + gateway_ip
localization_server = LocalizationServerClient(server_ip, server_port)

data = localization_client.get_measurement()

path = ""
serializer = JsonSerializer(path)
serializer.serialize(data)

localization_server.within_room(data)
