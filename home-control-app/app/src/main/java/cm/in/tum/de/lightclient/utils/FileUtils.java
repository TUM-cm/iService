package cm.in.tum.de.lightclient.utils;

import java.io.InputStream;

import cm.in.tum.de.lightclient.MainActivity;

public class FileUtils {

    public static InputStream getInputStream(int resource) {
        return MainActivity.getInstance().getResources().openRawResource(resource);
    }

    public static String getFileExtension(String filename) {
        String extension = "";
        int i = filename.lastIndexOf('.');
        if (i > 0) {
            extension = filename.substring(i);
        }
        return extension;
    }

}
