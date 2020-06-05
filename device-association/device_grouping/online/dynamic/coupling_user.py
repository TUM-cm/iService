import numpy
import random
import threading
from collections import deque
from coupling.device_grouping.online.dynamic.coupling_data_provider import CouplingDataProvider

class CouplingUser:
    
    def __init__(self, identifier, route, room_distances, client_data, scheduling_prio=1):
        self.identifier = identifier
        self.save_route = route
        self.route = deque(route) # [(duration_1,room_1), ..., (duration_n,room_n)]
        single_client_data = random.choice(client_data.values())
        self.coupling_data_provider = CouplingDataProvider(single_client_data)
        self.client_data = client_data
        self.room_distances = room_distances
        self.scheduling_prio = scheduling_prio
        self.current_room = None
        self.next_room = self.route.popleft()
        self.client = None
    
    def set_client(self, client):
        self.client = client
    
    def start(self):
        self.arrived()
    
    def arrived(self):
        self.current_room = self.next_room
        self.coupling_data_provider.set_signal(self.client_data[self.current_room[1]])
        stay_duration = self.current_room[0]
        if len(self.route) > 0: # schedule next move
            thread = threading.Timer(stay_duration, self.move)
            thread.start()
        else:
            thread = threading.Timer(stay_duration, self.leave)
            thread.start()
    
    def leave(self):
        if self.client:
            self.client.disconnect()
    
    def move(self):
        self.next_room = self.route.popleft()
        moving_distance = self.room_distances[self.current_room[1]][self.next_room[1]]
        move_duration = moving_distance / self.__get_moving_speed()
        self.coupling_data_provider.set_random_signal()
        thread = threading.Timer(move_duration, self.arrived)
        thread.start()
    
    # Client chooses each time for a new transition between rooms a random speed in range of
    # 1.35 - 1.65 m/s, (4.86 - 5.94 km/h)     https://de.wikipedia.org/wiki/Schrittgeschwindigkeit#cite_note-2
    # 1.25 - 1.53 m/s (4.5 - 5.5 km/h)        https://de.wikipedia.org/wiki/Gehen#cite_note-1
    def __get_moving_speed(self):
        return round(random.choice(numpy.arange(1.25, 1.53, 0.01)), 2)
    
    def get_coupling_data_provider(self):
        return self.coupling_data_provider
    
    def get_route(self):
        return self.save_route
    