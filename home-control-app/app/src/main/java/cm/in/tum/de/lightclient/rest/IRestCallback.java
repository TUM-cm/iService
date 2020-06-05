package cm.in.tum.de.lightclient.rest;

public interface IRestCallback {

  void onGetToken(String token);
  void onGetEncryptedData(String encryptedData);

}
