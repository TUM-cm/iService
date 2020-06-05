from enum import Enum

class PacketType(bytes, Enum):
    
    Query_Raw_Light = b'\x01'
    Response_Raw_Light = b'\x11'
    
    Query_Pattern_Light = b'\x02'
    Response_Light_Pattern = b'\x22'
    Response_Light_Pattern_Duration = b'\x23'
    Response_Light_Pattern_Signal = b'\x24'
    
    Query_Raw_WiFi = b'\x03'
    Response_Raw_WiFi = b'\x13'
    Query_Raw_BLE = b'\x04'
    Response_Raw_BLE = b'\x14'
