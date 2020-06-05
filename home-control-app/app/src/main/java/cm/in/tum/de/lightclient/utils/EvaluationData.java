package cm.in.tum.de.lightclient.utils;

import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class EvaluationData {

  private final static String TAG = EvaluationData.class.getSimpleName();
  private final List<AuthenticationResult> authenticationResults;
  private final DateTime dateTime;

  public EvaluationData() {
    this.authenticationResults = new ArrayList<>();
    this.dateTime = new DateTime();
  }

  public void saveAuthentication() {
    try {
      String filename = "authentication.txt";
      String path = AppStorage.STORAGE_PATH + "_" +
              getDateTime().getCurrentDateTime() + "_" +
              filename;
      BufferedWriter writer = new BufferedWriter(new FileWriter(path));
      Gson gson = new GsonBuilder().setPrettyPrinting().create();
      gson.toJson(getAuthenticationResults(), writer);
      writer.flush();
      writer.close();
    } catch (IOException e) {
      Log.d(TAG, "write evaluation", e);
    }
  }

  public void addAuthenticationResult(long duration, boolean result) {
    getAuthenticationResults().add(new AuthenticationResult(duration, result));
  }

  private List<AuthenticationResult> getAuthenticationResults() {
    return this.authenticationResults;
  }

  private DateTime getDateTime() {
    return this.dateTime;
  }

  class AuthenticationResult {

    private final long duration;
    private final boolean result;

    public AuthenticationResult(long duration, boolean result) {
      this.duration = duration;
      this.result = result;
    }

  }

}
