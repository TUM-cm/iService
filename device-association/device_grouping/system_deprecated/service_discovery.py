import re
import base64
import logging
import hashlib
from crypto.aes import AESCipher
try:
    from send_light.send_light_user import LightSender
except:
    pass

# message format
#     location identifier len: 1 byte (1 octet = tupel of 8 bit)
#     location identifier: 0-32 bytes (check minimum SSID len)
#     password len: 1 byte
#     password: 8-63 bytes
#     len services: 1 byte
#         service name len: 1 byte
#         service name: 1-32 bytes
#         service description len: 1 byte
#         service description: 140 bytes

def fill_len(length):
    return "{:03}".format(length)

def filter_input(data, cond=r'[^A-Za-z0-9]'):
    return re.sub(cond, '', data)

class Service:
    
    def __init__(self, name, description):
        self.name = filter_input(name)
        self.len_name = len(self.name)
        self.description = filter_input(description)
        self.len_description = len(self.description)
        assert 1 <= self.len_name <= 32
        assert 1 <= self.len_description <= 140
    
    def get_str(self):
        msg = []
        msg.append(fill_len(self.len_name))
        msg.append(self.name)
        msg.append(fill_len(self.len_description))
        msg.append(self.description)
        return "".join(msg)
    
    def __repr__(self):
        return "name: {0}, description: {1}".format(self.name, self.description)
    
class MessageServiceDiscovery:
    
    # location id = SSID of Wi-Fi access point
    def construct(self, location_id, password, services):
        self.aes, self.aes_password = AESCipher.keyFromGen(base64.b16encode, base64.b16decode)
        self.location_id = filter_input(location_id)
        self.len_location_id = len(self.location_id)
        self.password = filter_input(password) # 8-63 bytes in WPA-PSK mode
        self.len_password = len(self.password)
        self.services = "".join([service.get_str() for service in services])
        try:
            self.light_sender = LightSender()
        except:
            pass
        assert 1 <= self.len_location_id <= 20
        assert 8 <= self.len_password <= 63
    
    # data integrity
    def __generate_hash(self, msg):
        hash_func = hashlib.sha1()
        hash_func.update(msg)
        return hash_func.hexdigest()
    
    def get_message(self):
        msg = []
        msg.append(fill_len(self.len_location_id))
        msg.append(self.location_id)
        msg.append(fill_len(self.len_password))
        msg.append(self.password)
        msg.append(self.services)
        msg_hash = self.__generate_hash("".join(msg))
        msg.append(fill_len(len(msg_hash)))
        msg.append(msg_hash)
        msg = self.aes.encrypt("".join(msg))
        return self.aes_password + msg
    
    def broadcast(self):
        msg = self.get_message()
        self.light_sender.set_data(msg)
        self.light_sender.start()
    
    def parse(self, msg, end=0, len_len=3, pwd_len=32, min_pkt_size=250):
        if len(msg) < min_pkt_size:
            return
        msg = msg.strip().upper()
        # decrypt
        key = msg[:pwd_len].lower()
        self.aes = AESCipher.keyFromVariable(key, base64.b16encode, base64.b16decode)
        msg = self.aes.decrypt(msg[pwd_len:])
        # parse msg
        self.hash = None
        self.msg = list()
        self.data = list()
        self.services = list()
        len_msg = len(msg)
        while end < len_msg:
            start = end
            end += len_len
            len_data = msg[start:end]
            start = end
            end += int(len_data)
            temp_data = msg[start:end]
            if end == len_msg:
                self.hash = temp_data
            else:
                self.msg.append(len_data)
                self.msg.append(temp_data)
                self.data.append(temp_data)
        # create services
        index = 2
        data_len = len(self.data)
        while index < data_len:
            service = Service(self.data[index], self.data[index+1])
            self.services.append(service)
            index += 2
    
    def get_location(self):
        if self.data and len(self.data) > 1:
            return self.data[0]
    
    def get_password(self):
        if self.data and len(self.data) > 1:
            return self.data[1]
    
    def get_services(self):
        return self.services
    
    def is_valid(self):
        msg = "".join(self.msg)
        logging.debug(msg)
        hash_func = hashlib.sha1()
        hash_func.update(msg)
        return (hash_func.hexdigest() == self.hash)
    
def test_msg():
    services = [Service("a", "abc"), Service("b", "def"), Service("c", "123")]
    msg = MessageServiceDiscovery()
    msg.construct("01.05.038", "dummydummy", services)
    return msg.get_message()
    
def test_parse_msg(test_msg):
    service_data = MessageServiceDiscovery()
    service_data.parse(test_msg)
    print(service_data.get_location())
    print(service_data.get_password())
    print(service_data.get_services())
    print(service_data.is_valid())

def test_light_sender(test_data):
    light_sender = LightSender()
    light_sender.set_data(test_data)    
    light_sender.start()
    
def test():
    logging.basicConfig(level=logging.DEBUG)
    msg = test_msg()
    print(msg)
    test_parse_msg(msg)
    
if __name__ == "__main__":
    test()
