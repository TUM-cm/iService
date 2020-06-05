package cm.in.tum.de.lightclient.utils;

import android.util.Log;

import org.ini4j.Ini;

import java.io.IOException;
import java.io.InputStream;

public class Config {

	private static final String TAG = Config.class.getSimpleName();
	private Ini ini;

	public Config(int resource) {
		try {
			InputStream inputStream = FileUtils.getInputStream(resource);
			this.ini = new Ini(inputStream);
		} catch (IOException e) {
			Log.e(TAG, "Config resource", e);
		}
	}

	private Ini getIni() {
		return this.ini;
	}

	public <T> T get(String key, String section, Class<T> type) {
		return getIni().get(section).get(key, type);
	}

}
