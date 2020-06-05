from __future__ import division
import os
import copy
import glob
import numpy
import random
import datetime
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import coupling.utils.misc as misc
from collections import OrderedDict
from collections import defaultdict
from datetimerange import DateTimeRange
from utils.nested_dict import nested_dict
from utils.serializer import DillSerializer
from matplotlib.offsetbox import AnchoredText
from utils.nested_dict import nested_ordered_dict
from coupling.utils.defaultordereddict import DefaultOrderedDict

class DateTimeRangeHashable(DateTimeRange):
    
    def __hash__(self):
        return hash(repr(self))
    
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
setting_log_testbeds = os.path.join(__location__, "log-data", "setting-log-testbeds")

class DeviceClass:
    
    def __init__(self, name, name_abbr, allowed_encounter_times):
        self.name = name
        self.name_abbr = name_abbr
        self.allowed_encounter_times = allowed_encounter_times

device_class_personal = DeviceClass("personal", "p", "morning[workdays],evening[workdays],noon[all],afternoon[all],evening[all]")
device_class_family = DeviceClass("family & friends", "f&f", "morning[workdays],evening[workdays],all[weekends,holidays]")
device_class_stranger = DeviceClass("well-known & stranger", "wk&s", "all[all]")
device_classes = [device_class_personal, device_class_family, device_class_stranger]

class TestbedSetting:
    
    def __init__(self, sampling_resolution, min_coupling_time, max_coupling_time,
                 time_encounter_ratio_personal, num_devices_personal,
                 time_encounter_ratio_family_friends, num_devices_family_friends,
                 time_encounter_ratio_well_known_stranger, num_devices_well_known_stranger,
                 columns=["time encounter ratio", "allowed encounter times", "num. devices"], to_seconds=60):
        
        self.min_coupling_time = min_coupling_time
        self.max_coupling_time = max_coupling_time
        self.sampling_resolution = sampling_resolution * to_seconds
        self.log_distribution = nested_dict(2, dict)
        
        testbed_device_classes = {device_class_personal.name: (time_encounter_ratio_personal, num_devices_personal),
                                  device_class_family.name: (time_encounter_ratio_family_friends, num_devices_family_friends),
                                  device_class_stranger.name: (time_encounter_ratio_well_known_stranger, num_devices_well_known_stranger)}    
        for device_class in device_classes:
            testbed_device_class = testbed_device_classes[device_class.name]
            self.set_data(self.log_distribution, device_class.name, columns,
                          [testbed_device_class[0], device_class.allowed_encounter_times, testbed_device_class[1]])
    
    def set_data(self, data, row, columns, values):
        for column, value in zip(columns, values):
            data[row][column] = value

def autolabel(ax, bars, labels):
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
                1.01 * height, label, ha="center", va="bottom")

def get_device_type(mac, device_pool):
    for device_type, macs in device_pool.items():
        if mac in macs:
            return device_type
    return None

def get_data_path(identifier, paths):
    for path in paths:
        if identifier in os.path.basename(path):
            return path

def get_log_distribution(coupling_log, sampling_resolution):
    log_per_day = DefaultOrderedDict(list)
    log_per_datetime = OrderedDict()
    log_per_day_datetime = nested_ordered_dict(2, dict)
    all_encounters = coupling_log.keys()
    timedelta = datetime.timedelta(seconds=sampling_resolution)
    for relative_day, (_, encounters) in enumerate(itertools.groupby(all_encounters, key=lambda date: date.day)):
        encounters = list(encounters) # convert generator to list otherwise iterate only once
        slot_split_idx = list()
        for i, (d1, d2) in enumerate(misc.pairwise(encounters)):
            if (d2 - d1) != timedelta:
                slot_split_idx.append(i)
        slot_split_idx = numpy.asarray(slot_split_idx)
        for i, time_ranges in enumerate(numpy.split(encounters, slot_split_idx+1)):
            devices = [coupling_log[time_range] for time_range in time_ranges]
            devices = list(set(misc.flatten_list(devices)))
            datetime_start = time_ranges[0]
            # device seen only once, at one timestamp: set to minimum duration: sampling resolution
            datetime_stop = datetime_start + timedelta if len(time_ranges) == 1 else time_ranges[-1]
            time_range = DateTimeRangeHashable(datetime_start, datetime_stop)
            log_per_datetime[time_range] = devices
            day = relative_day + 1
            log_per_day[day].append(
                (datetime_start, datetime_stop, devices))
            log_per_day_datetime[day][time_range] = devices        
    return log_per_day, log_per_datetime, log_per_day_datetime

class SimulationCalendar:
    
    def __init__(self, num_weeks, num_holidays, sampling_resolution):
        self.days_per_week = 7
        self.num_days = num_weeks * self.days_per_week
        self.days = range(1, self.num_days+1)
        today = datetime.datetime.today()
        self.start_day = datetime.datetime(today.year, today.month, 1, 0, 0, 0)
        self.end_day = self.start_day + datetime.timedelta(days=self.num_days)
        self.current_day = self.start_day
        self.timedelta = datetime.timedelta(seconds=sampling_resolution)
        self.day_distribution = dict()
        self.day_distribution["all"] = self.days
        self.day_distribution["holidays"] = random.sample(self.days, num_holidays)
        self.day_distribution["weekends"] = [day for day in self.days if self.is_weekend(day)]
        self.day_distribution["workdays"] = [day for day in self.days if self.is_workday(day)]
        self.time_distribution = dict()
        self.time_distribution["all"] = [0, 24]
        self.time_distribution["night morning"] = [0, 5]
        self.time_distribution["morning"] = [5, 10]
        self.time_distribution["forenoon"] = [10, 12]
        self.time_distribution["noon"] = [12, 14]
        self.time_distribution["afternoon"] = [14, 17]
        self.time_distribution["evening"] = [17, 21]
        self.time_distribution["evening night"] = [21, 24]
    
    def get_days(self):
        return numpy.arange(1, (self.end_day)+1)
    
    def get_total_time(self):
        return self.num_days * 24 * 60
    
    def is_weekend(self, day):
        day_per_week = day % self.days_per_week
        if day_per_week == 6 or day_per_week == 0:
            return True
        else:
            return False
    
    def is_holiday(self, day):
        return day in self.day_distribution["holidays"]
    
    def is_weekend_or_holiday(self, day):
        return self.is_weekend(day) or self.is_holiday(day)
    
    def is_workday(self, day):
        return not self.is_weekend_or_holiday(day)
    
    def get_day_distribution(self, day_identifier):
        return self.day_distribution[day_identifier]
    
    def get_time_distribution(self, time_identifier):
        return self.time_distribution[time_identifier]
    
    def get_timedelta(self):
        return self.timedelta
    
def create_device_pool(data_device_coupling):
    device_pool = dict()
    for device_type, data in data_device_coupling.items():
        other_macs = misc.flatten_list(device_pool.values())
        duplicate_mac = [True]
        while True in duplicate_mac:
            macs = [misc.create_random_mac() for _ in range(data["num. devices"])]
            duplicate_mac = [mac in other_macs for mac in macs]
        device_pool[device_type] = macs
    return device_pool

def create_coupling_log(data_device_coupling, device_pool, simulation_durations, total_holidays,
                        sampling_resolution, min_coupling_time, max_coupling_time, weeks_per_year=52):
    
    def merge_time_slots(time_slots):
        merge_time_slots = dict()
        for day, time_slots_day in time_slots.items():
            times = copy.deepcopy(time_slots_day)
            times.sort()
            times = list(times for times, _ in itertools.groupby(times))
            slots = [times[0]]
            for slot in times[1:]:
                if slots[-1][1] == slot[0]: # join
                    slots[-1][1] = slot[1]
                else:
                    slots += [slot]
            merge_time_slots[day] = slots
        return merge_time_slots
    
    coupling_logs = dict()
    today = datetime.datetime.today()
    simulation_start = datetime.datetime(today.year, today.month, 1, 0, 0, 0)
    for num_weeks in simulation_durations:
        coupling_log = defaultdict(list)
        num_holidays = int(num_weeks * (total_holidays / weeks_per_year))
        print("num weeks: ", num_weeks)
        print("num holidays: ", num_holidays)
        calendar = SimulationCalendar(num_weeks, num_holidays, sampling_resolution)
        device_time_slot_distribution = nested_dict(2, list)
        for device_type, data in data_device_coupling.items():
            total_encounter_time = 0
            time_slot_distribution = defaultdict(list)
            for allowed_encounter_times in data["allowed encounter times"].split("],"):
                split_idx = allowed_encounter_times.index("[")
                semantic_time = allowed_encounter_times[:split_idx]
                end_idx = -1 if "]" in allowed_encounter_times else len(allowed_encounter_times)
                semantic_days = allowed_encounter_times[split_idx+1:end_idx].split(",")
                time_interval = calendar.get_time_distribution(semantic_time)
                encounter_time = time_interval[1] - time_interval[0]
                for semantic_day in semantic_days:
                    days = calendar.get_day_distribution(semantic_day)
                    for day in days:
                        time_slot_distribution[day].append(time_interval)    
                num_days = sum([len(calendar.get_day_distribution(day)) for day in semantic_days])
                total_encounter_time += (encounter_time * num_days) * 60 # minutes
            
            time_slot_distribution = merge_time_slots(time_slot_distribution)
            start_total_time = int(data["time encounter ratio"][0] * total_encounter_time)
            end_total_time = int(data["time encounter ratio"][1] * total_encounter_time)
            total_encounter_time = random.randint(start_total_time, end_total_time)
            print("device type: ", device_type)
            while total_encounter_time > min_coupling_time: # distribute encounter time over weeks
                # equivalent:
                #numpy.random.choice(range(min_coupling_time, max_coupling_time, sampling_resolution))
                coupling_time = random.randrange(
                    min_coupling_time, min(max_coupling_time, total_encounter_time), sampling_resolution)
                coupling_time = datetime.timedelta(minutes=coupling_time)
                day = int(numpy.random.choice(list(time_slot_distribution.keys())))
                time_slot = int(numpy.random.choice(len(time_slot_distribution[day])))
                time_slot = time_slot_distribution[day][time_slot] # select actual range
                day = simulation_start + datetime.timedelta(days=day-1)
                start_time = day.replace(hour=time_slot[0])
                if time_slot[1] == 24:
                    end_time = day.replace(hour=time_slot[1]-1, minute=55) - coupling_time
                else:
                    end_time = day.replace(hour=time_slot[1]) - coupling_time
                start_slot_range = list()
                start_slot_range.append(start_time)
                while start_slot_range[-1] + calendar.get_timedelta() <= end_time:
                    start_slot_range.append(start_slot_range[-1] + calendar.get_timedelta())
                start_time = numpy.random.choice(start_slot_range)
                end_time = start_time + coupling_time
                devices = device_pool[device_type]
                num_devices = random.randint(1, len(devices))
                devices = random.sample(devices, num_devices)
                coupling_duration = end_time - start_time
                num_samples = int(coupling_duration.total_seconds() / calendar.get_timedelta().total_seconds())
                log_time_range = [start_time + i * calendar.get_timedelta() for i in range(num_samples+1)]
                for log_time in log_time_range:
                    coupling_log[log_time].extend(devices)
                total_encounter_time -= int(coupling_time.total_seconds() / 60)
                device_time_slot_distribution[day][device_type].append((start_time, end_time))
        
        # remove single devices and duplicates at specific points in time
        clean_coupling_log = dict()
        for timestamp, devices in coupling_log.items():
            unique_devices = list(set(devices))
            if len(unique_devices) > 1:
                clean_coupling_log[timestamp] = unique_devices
        coupling_logs[num_weeks] = OrderedDict(sorted(clean_coupling_log.items(), key=lambda t: t[0]))
    return coupling_logs

def plot_coupling_log(num_weeks, coupling_log, sampling_resolution, device_pool, result_directory, plot_format):
    
    def plot_calendar_bars(num_weeks, days, event_start, duration, device_group_labels,
                  date_anchor=datetime.datetime(1,1,1,0,0,0), sampling_days=14, sampling_hours=6):
        print("plot calendar")
        all_days = range(1, 1+num_weeks*7)
        diff_days = set(all_days).difference(days)
        if len(diff_days) > 0:
            diff_days_idx = [numpy.where(days < diff_day)[0][-1]+1 for diff_day in diff_days]
            days = numpy.insert(days, diff_days_idx, list(diff_days))
            duration = numpy.insert(duration, diff_days_idx, [0]*len(diff_days_idx))
            event_start = numpy.insert(
                event_start, diff_days_idx, [mpl.dates.date2num(date_anchor)]*len(diff_days_idx))
        fig, ax = plt.subplots()
        xpos = days / 3
        device_groups = list()
        rect_device_groups = list()
        device_group_labels = numpy.asarray(device_group_labels)
        unique_device_groups = set(device_group_labels)
        colormap = plt.cm.tab20
        colors = [colormap(i) for i in numpy.linspace(0, 1, len(unique_device_groups))]
        for i, device_group in enumerate(unique_device_groups):
            groupidx = numpy.where(device_group_labels == device_group)
            rects = ax.bar(x=xpos[groupidx], height=duration[groupidx],
                           bottom=event_start[groupidx], width=0.1,
                           color=colors[i], label=device_group)
            rect_device_groups.append(rects)
            device_groups.append(device_group)
        xtickpos = numpy.asarray(sorted(list(set(xpos))))
        xtickdays = numpy.asarray(sorted(list(set(days))))
        xtickidx = numpy.where(xtickdays % sampling_days == 0)
        xtickidx = numpy.insert(xtickidx, 0, 0)
        xtickidx = xtickidx[::2]
        
        ax.set_xticks(xtickpos[xtickidx])
        ax.set_xticklabels(xtickdays[xtickidx])
        yticks = [mpl.dates.date2num(date_anchor.replace(hour=hour)) for hour in range(24)]
        yticks.append(mpl.dates.date2num(date_anchor.replace(hour=23, minute=59)))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_yticks(yticks[::sampling_hours])
        ax.set_xlabel("Days")
        ax.set_ylabel("Time")
        ax.yaxis.grid()
        fig.set_figwidth(fig.get_figwidth()*2)
        filename = "coupling-log-calendar-num-weeks-" + str(num_weeks) + "." + plot_format
        filepath = os.path.join(result_directory, filename)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        fig_legend = plt.figure(figsize = (20, 2))
        plt.figlegend(*ax.get_legend_handles_labels(), loc="center", ncol=3)
        filename = "coupling-log-calendar-num-weeks-" + str(num_weeks) + "-legend." + plot_format
        filepath = os.path.join(result_directory, filename)
        print(filepath)
        fig_legend.savefig(filepath, format=plot_format, bbox_inches="tight")
        #plt.show()
        plt.close(fig)
        plt.close(fig_legend)
    
    def plot_device_bars(data, ylabel, bar_width=0.2):
        device_class_abbr = {device_class.name: device_class.name_abbr for device_class in device_classes}
        print("plot devices")
        fig, ax = plt.subplots()
        ind = numpy.arange(len(data))
        bars = ax.bar(ind, [device[0] for device in data.values()], bar_width)
        labels = [device_class_abbr[device[1]] for device in data.values()]
        autolabel(ax, bars, labels)
        anchored_text = "      ".join(abbr + ": " + full for full, abbr in device_class_abbr.items())
        at = AnchoredText(anchored_text, loc="lower left", frameon=True,
                          bbox_to_anchor=(0., 1.), bbox_transform=ax.transAxes)
        at.patch.set_boxstyle("round,pad=0.")
        ax.add_artist(at)    
        ax.set_xticks(ind)
        ax.set_xticklabels(data.keys(), rotation=30, ha="right")
        ax.set_xlabel("Devices")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid()
        filename = ylabel.replace(" ", "-").lower()
        filepath = os.path.join(
            result_directory, filename + "." + plot_format)
        print(filepath)
        fig.savefig(filepath, format=plot_format, bbox_inches="tight")
        #plt.show()
        plt.close(fig)
    
    log_per_day, log_per_time, _ = get_log_distribution(coupling_log, sampling_resolution)
    days = numpy.asarray(misc.flatten_list([[day] * len(events) for day, events in log_per_day.items()]))
    device_group_labels = list()
    event_start = numpy.empty(len(days))
    event_finish = numpy.empty(len(days))
    i = 0
    for events in log_per_day.values():
        for start, end, devices in events:
            # reset date and convert to numeric value
            event_start[i] = mdates.date2num(start.replace(year=1, month=1, day=1))
            event_finish[i] = mdates.date2num(end.replace(year=1, month=1, day=1))
            device_types = list(set([get_device_type(device, device_pool) for device in devices]))
            device_group_labels.append("-".join(device_types))
            i += 1
    duration = event_finish - event_start
    plot_calendar_bars(num_weeks, days, event_start, duration, device_group_labels)
    
    devices = misc.flatten_list(coupling_log.values())
    unique_devices = list(set(devices))
    device_contact_frequency = dict()
    for device in unique_devices:
        frequency = [device in devices for devices in coupling_log.values()]
        contact_frequency = frequency.count(True) / len(coupling_log)
        device_contact_frequency[device] = (contact_frequency, get_device_type(device, device_pool))
    plot_device_bars(device_contact_frequency, "Ratio contact frequency")
    coupling_timestamps = list(coupling_log.keys())
    log_duration = coupling_timestamps[-1] - coupling_timestamps[0]
    total_log_time = log_duration.total_seconds() / 60
    device_time_coupling = dict()
    for device in unique_devices:
        time_ranges = [time_range for time_range, devices in log_per_time.items() if device in devices]
        time_coupling = sum([time_range.get_timedelta_second() for time_range in time_ranges]) / 60
        device_time_coupling[device] = (time_coupling / total_log_time, get_device_type(device, device_pool))
    plot_device_bars(device_time_coupling, "Ratio time coupling")

def plot_calendar_and_devices(result_directory):
    plot_format = "pdf"
    setting_testbeds = DillSerializer(setting_log_testbeds).deserialize()
    log_data_folder = os.path.dirname(setting_log_testbeds)
    testbeds = ["full", "mid", "sparse"]
    coupling_log_paths = misc.flatten_list([glob.glob(os.path.join(log_data_folder, testbed + "*coupling-log*")) for testbed in testbeds])
    device_pool_paths = misc.flatten_list([glob.glob(os.path.join(log_data_folder, testbed + "*device-pool*")) for testbed in testbeds])
    for path_log_data, path_device_pool in zip(coupling_log_paths, device_pool_paths):
        print("path log data: ", path_log_data)
        print("path device pool: ", path_device_pool)
        testbed = os.path.basename(path_log_data).split("-")[0]
        assert testbed == os.path.basename(path_device_pool).split("-")[0]
        print("testbed: ", testbed)    
        settings = setting_testbeds[testbed]
        device_pool = DillSerializer(path_device_pool).deserialize()
        coupling_logs = DillSerializer(path_log_data).deserialize()
        for num_weeks in sorted(coupling_logs):
            print("num weeks: ", num_weeks)
            result_plot = os.path.join(result_directory, testbed)
            if not os.path.exists(result_plot):
                os.makedirs(result_plot)
            plot_coupling_log(
                num_weeks, coupling_logs[num_weeks], settings.sampling_resolution,
                device_pool, result_plot, plot_format)
        print("---")
    
def process_coupling_log():
    total_holidays = 13
    log_testbeds = {"sparse": TestbedSetting(5, 10, 60, (0.4, 0.5), 3, (0.3, 0.4), 3, (0.05, 0.1), 3),
                    "mid": TestbedSetting(5, 20, 120, (0.5, 0.6), 6, (0.3, 0.4), 6, (0.05, 0.1), 6),
                    "full": TestbedSetting(5, 30, 180, (0.6, 0.7), 9, (0.3, 0.4), 9, (0.05, 0.1), 9)}
    
    log_data = os.path.dirname(setting_log_testbeds)
    if not os.path.exists(log_data):
        os.makedirs(log_data)
    
    simulation_durations = [52] # weeks
    DillSerializer(setting_log_testbeds).serialize(log_testbeds)
    for description, testbed_setting in log_testbeds.items():
        print("### testbed: ", description)
        device_pool = create_device_pool(testbed_setting.log_distribution)
        path_device_pool = os.path.join(log_data, description + "-device-pool")
        DillSerializer(path_device_pool).serialize(device_pool)
        coupling_log = create_coupling_log(
            testbed_setting.log_distribution, device_pool, simulation_durations, total_holidays,
            testbed_setting.sampling_resolution, testbed_setting.min_coupling_time, testbed_setting.max_coupling_time)
        path_single_coupling_log = os.path.join(log_data, description + "-coupling-log")
        DillSerializer(path_single_coupling_log).serialize(coupling_log)
        print("------------")

def main():
    process_coupling_log()
    
if __name__ == "__main__":
    main()
    