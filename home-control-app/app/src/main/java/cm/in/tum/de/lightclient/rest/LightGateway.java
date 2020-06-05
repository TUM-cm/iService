package cm.in.tum.de.lightclient.rest;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;

public interface LightGateway {

  @GET("token")
  Call<String> getToken();

  @GET("encryption/{data}")
  Call<String> getEncryptedData(@Path("data") String data);

}
