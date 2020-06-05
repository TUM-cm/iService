import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# https://stackoverflow.com/questions/4790265/plot-time-of-day-vs-date-in-matplotlib
def plot_time_points():
    # Make a series of events 1 day apart
    x = mpl.dates.drange(datetime.datetime(2009,10,1), 
                         datetime.datetime(2010,1,15), 
                         datetime.timedelta(days=1))
    # Vary the datetimes so that they occur at random times
    # Remember, 1.0 is equivalent to 1 day in this case...
    x += np.random.random(x.size)
    # We can extract the time by using a modulo 1, and adding an arbitrary base date
    times = x % 1 + int(x[0]) # (The int is so the y-axis starts at midnight...)
    # I'm just plotting points here, but you could just as easily use a bar.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot_date(x, times, 'ro')
    ax.yaxis_date()
    fig.autofmt_xdate()
    plt.show()

# https://stackoverflow.com/questions/2207670/date-versus-time-interval-plotting-in-matplotlib
def plot_time_bars():
    # dates for xaxis
    event_date = [datetime.datetime(2008, 12, 3), datetime.datetime(2009, 1, 5), datetime.datetime(2009, 2, 3)]
    # base date for yaxis can be anything, since information is in the time
    anydate = datetime.date(2001,1,1)
    # event times
    event_start = [datetime.time(20, 12), datetime.time(12, 15), datetime.time(8, 1,)]
    event_finish = [datetime.time(23, 56), datetime.time(16, 5), datetime.time(18, 34)]
    # translate times and dates lists into matplotlib date format numpy arrays
    start = np.fromiter((mdates.date2num(datetime.datetime.combine(anydate, event)) for event in event_start), dtype = 'float', count = len(event_start))
    finish = np.fromiter((mdates.date2num(datetime.datetime.combine(anydate, event)) for event in event_finish), dtype = 'float', count = len(event_finish))
    date = mdates.date2num(event_date)
    # calculate events durations
    duration = finish - start
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # use errorbar to represent event duration
    ax.errorbar(date, start, [np.zeros(len(duration)), duration], linestyle='')
    # make matplotlib treat both axis as times
    ax.xaxis_date()
    ax.yaxis_date()
    plt.show()

def main():
    plot_time_bars()
    plot_time_points()
    
if __name__ == "__main__":
    main()
