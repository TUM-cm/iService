package cm.in.tum.de.lightclient.utils;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DateTime {

  private static final String OUTPUT_PATTERN = "yyyy_MM_dd_HH_mm_ss";
  private final DateFormat dateFormat;

  public DateTime() {
    this.dateFormat = new SimpleDateFormat(OUTPUT_PATTERN);
  }

  public String getCurrentDateTime() {
    Date date = new Date(getUnixTimestamp() * 1000L);
    return getDateFormat().format(date);
  }

  public long getUnixTimestamp() {
    return (System.currentTimeMillis() / 1000L);
  }

  private DateFormat getDateFormat() {
    return this.dateFormat;
  }

}
