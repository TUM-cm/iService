package cm.in.tum.de.lightclient.utils;

import android.os.Environment;

import java.io.File;

public class AppStorage {

  public static final String STORAGE_FOLDER_NAME = "ClientHomeControl";
  public static final String STORAGE_PATH = Environment.getExternalStorageDirectory().getAbsolutePath() +
          File.separator + STORAGE_FOLDER_NAME + File.separator;

  public static void createDataFolder() {
    File directory = new File(AppStorage.STORAGE_PATH);
    if (!directory.exists()) {
      directory.mkdirs();
    }
  }

}
