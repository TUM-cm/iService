package cm.in.tum.de.lightclient.utils;

import android.os.SystemClock;

public class StopWatch {

  private long startTime;
  private long stopTime;
  private boolean running;

  public StopWatch() {
    this.startTime = 0;
    this.stopTime = 0;
    this.running = false;
  }

  public void start() {
    this.startTime = SystemClock.elapsedRealtime();
    this.running = true;
  }

  public void stop() {
    this.stopTime = SystemClock.elapsedRealtime();
    this.running = false;
  }

  public long getElapsedTime() {
    long elapsed;
    if (this.running) {
      elapsed = (SystemClock.elapsedRealtime() - startTime);
    } else {
      elapsed = (stopTime - startTime);
    }
    return elapsed;
  }

}
