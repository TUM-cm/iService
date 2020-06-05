import numpy
import random
try:
    from send_light.send_light_user import LightSender
except:
    pass

class LightPattern:
    
    # min duration: limit by sampling rate
    # max duration: limit by human perception to see light signal
    def __init__(self, min_duration=1000000, max_duration=5000000, min_len_pattern=4, max_len_pattern=10):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_len_pattern = min_len_pattern
        self.max_len_pattern = max_len_pattern
        try:
            self.light_sender = LightSender()
        except:
            pass
    
    # insmod send_light_kernel.ko action=pattern pattern=10,20,10,20,...    
    def broadcast(self, pattern_len=None):
        pattern = self.get_pattern(pattern_len)
        #pattern = ",".join(map(str, pattern))
        self.light_sender.set_pattern_series(pattern)
        self.light_sender.start()
        return pattern
    
    def get_pattern(self, pattern_len=None):
        if not pattern_len: # Random pattern length between min and max, multiple of two
            pattern_len = 1
            while pattern_len % 2 != 0:
                pattern_len = random.randint(self.min_len_pattern, self.max_len_pattern)
        phase_len = pattern_len//2
        on_duration = self.__gen_values(phase_len)
        off_duration = self.__gen_values(phase_len)
        return [i for j in zip(on_duration, off_duration) for i in j]
    
    def __too_similar(self, new_value, values, tolerance=0.1):
        return numpy.any(numpy.isclose(new_value, values, rtol=tolerance))
    
    def __gen_values(self, phase_len):
        values = []
        while len(values) != phase_len:
            new_value = numpy.random.randint(self.min_duration, self.max_duration)
            #new_value = numpy.random.randint(0, 2147483647)
            if not self.__too_similar(new_value, values):
                values.append(new_value)
        return values

def test_fixed_pattern(light_pattern):
    pattern_len = 4 # 2, 4, 6, 8, 10
    #pattern = light_pattern.get_pattern(pattern_len)
    light_pattern.broadcast((pattern_len))
    
def main():
    light_pattern = LightPattern()
    test_fixed_pattern(light_pattern)
    
if __name__ == "__main__":
    main()
    