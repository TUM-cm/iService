package cm.in.tum.de.lightclient;

import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import java.net.InetSocketAddress;

import cm.in.tum.de.lightclient.rest.IRestCallback;
import cm.in.tum.de.lightclient.rest.RestfulService;

public class RequestServer extends WebSocketServer implements IRestCallback {

  private RestfulService restfulService;
  private WebSocket connection;

  public RequestServer(InetSocketAddress address) {
    super(address);
  }

  public RequestServer() {
    super();
  }

  @Override
  public void onOpen(WebSocket conn, ClientHandshake handshake) {
    System.out.println("open");
  }

  @Override
  public void onClose(WebSocket conn, int code, String reason, boolean remote) {}

  @Override
  public void onMessage(WebSocket conn, String message) {
    if (message.contains("request_authentication")) {
      setConnection(conn);
      String challenge = message.split(":")[1];
      int nonce = Integer.valueOf(challenge);
      nonce++;
      getRestfulService().getEncryptedData(String.valueOf(nonce));
    } else if (message.contains("request_token")) {
      setConnection(conn);
      getRestfulService().getToken();
    }
  }

  @Override
  public void onError(WebSocket conn, Exception ex) {}

  @Override
  public void onStart() {}

  // Callbacks
  @Override
  public void onGetToken(String token) {
    getConnection().send(token);
  }

  @Override
  public void onGetEncryptedData(String encryptedData) {
    getConnection().send(encryptedData);
  }

  public void setRestfulService(RestfulService restfulService) {
    this.restfulService = restfulService;
  }

  private void setConnection(WebSocket connection) {
    this.connection = connection;
  }

  private WebSocket getConnection() {
    return this.connection;
  }

  private RestfulService getRestfulService() {
    return this.restfulService;
  }

}
