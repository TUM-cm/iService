package cm.in.tum.de.localvlc;

public interface IMqttListener {

  void onMqttConnect();
  void onMqttMessageReceived(String message);

}
