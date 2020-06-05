package cm.in.tum.de.lightclient.rest;

import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.IOException;

import cm.in.tum.de.lightclient.utils.Config;
import cm.in.tum.de.lightclient.utils.ConfigSection;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RestfulService {

  private static final String TAG = RestfulService.class.getSimpleName();

  private final IRestCallback restCallback;
  private final Config config;
  private final LightGateway lightGateway;
  private final HomeControlGateway homeControlGateway;

  public RestfulService(IRestCallback restCallback, String ipHomeControl, int configResource) {
    this.restCallback = restCallback;
    this.config = new Config(configResource);
    Retrofit.Builder lightGatewayBuilder = createRestBuilder(
            getConfigValue("Service Protocol", ConfigSection.LIGHT_RECEIVER),
            getConfigValue("Service IP", ConfigSection.LIGHT_RECEIVER),
            getConfigValue("Service Port", ConfigSection.LIGHT_RECEIVER),
            getConfigValue("Service Path", ConfigSection.LIGHT_RECEIVER));
    this.lightGateway = ServiceGenerator.createService(LightGateway.class,
            lightGatewayBuilder, true);
    Retrofit.Builder homeControlGatewayBuilder = createRestBuilder(
            getConfigValue("Service Protocol", ConfigSection.HOME_CONTROL),
            ipHomeControl,
            getConfigValue("Service Port", ConfigSection.HOME_CONTROL),
            getConfigValue("Service Path", ConfigSection.HOME_CONTROL));
    this.homeControlGateway = ServiceGenerator.createService(HomeControlGateway.class,
            homeControlGatewayBuilder, true);
  }

  private Retrofit.Builder createRestBuilder(String protocol, String host, String port, String servicePath) {
    Gson gson = new GsonBuilder().create();
    Retrofit.Builder builder = ServiceGenerator.createBuilder(
            createBaseUrl(protocol, host, port, servicePath),
            GsonConverterFactory.create(gson));
    return builder;
  }

  private String createBaseUrl(String protocol, String host, String port, String servicePath) {
    StringBuilder uriBuilder = new StringBuilder(protocol);
    uriBuilder.append("://");
    uriBuilder.append(host);
    uriBuilder.append(":");
    uriBuilder.append(port);
    uriBuilder.append(servicePath);
    return uriBuilder.toString();
  }

  //==========================================================================//
  // API
  //==========================================================================//
  public void getToken() {
    Call<String> call = getLightGateway().getToken();
    call.enqueue(new Callback<String>() {
      @Override
      public void onResponse(Call<String> call, Response<String> response) {
        if (response.isSuccessful()) {
          getRestCallback().onGetToken(response.body());
        } else {
          Log.e(TAG, "Failure get token");
        }
      }

      @Override
      public void onFailure(Call<String> call, Throwable t) {
        Log.e(TAG, "Failure get token", t);
      }
    });
  }

  public void getEncryptedData(String data) {
    Call<String> call = getLightGateway().getEncryptedData(data);
    call.enqueue(new Callback<String>() {
      @Override
      public void onResponse(Call<String> call, Response<String> response) {
        if (response.isSuccessful()) {
          getRestCallback().onGetEncryptedData(response.body());
        } else {
          Log.e(TAG, "Failure get encrypted data");
        }
      }

      @Override
      public void onFailure(Call<String> call, Throwable t) {
        Log.e(TAG, "Failure get encrypted data", t);
      }
    });
  }

  public boolean evaluateAuthenticaton() {
    try {
      Call<Boolean> call = getHomeControlGateway().evaluateAuthentication();
      Response<Boolean> response = call.execute();
      return response.body();
    } catch (IOException e) {
      Log.e(TAG, "Failure evaluate authentication");
    }
    return false;
  }

  private LightGateway getLightGateway() {
    return this.lightGateway;
  }

  private HomeControlGateway getHomeControlGateway() {
    return this.homeControlGateway;
  }

  private Config getConfig() {
    return this.config;
  }

  private String getConfigValue(String key, String section) {
    return getConfig().get(key, section, String.class);
  }

  private IRestCallback getRestCallback() {
    return this.restCallback;
  }

}
