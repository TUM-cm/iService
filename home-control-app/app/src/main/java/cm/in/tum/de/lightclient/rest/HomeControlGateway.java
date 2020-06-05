package cm.in.tum.de.lightclient.rest;

import retrofit2.Call;
import retrofit2.http.GET;

public interface HomeControlGateway {

  @GET("authentication")
  Call<Boolean> evaluateAuthentication();

}
