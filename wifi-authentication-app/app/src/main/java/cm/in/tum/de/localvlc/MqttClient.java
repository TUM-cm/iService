package cm.in.tum.de.localvlc;

import android.content.Context;
import android.util.Log;

import org.eclipse.paho.android.service.MqttAndroidClient;
import org.eclipse.paho.client.mqttv3.IMqttActionListener;
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.IMqttToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;

public class MqttClient {

  private static final String TAG = MqttClient.class.getSimpleName();
  private static final String MQTT_SERVER_PORT = ":1883";
  private static final String MQTT_SERVER_PROTOCOL = "tcp://";
  private static final int MQTT_QOS = 0;

  private final MqttAndroidClient mqttClient;
  private final MqttConnectOptions mqttConnectOptions;
  private final IMqttListener callback;
  private final String mqttServerUri;
  private final String clientId;

  public MqttClient(Context context, String serverIp, String clientId, IMqttListener callback) {
    this.clientId = clientId;
    this.callback = callback;
    this.mqttServerUri = MQTT_SERVER_PROTOCOL + serverIp + MQTT_SERVER_PORT;
    this.mqttClient = new MqttAndroidClient(context, getMqttServerUri(), getClientId());
    this.mqttConnectOptions = new MqttConnectOptions();
    getMqttClient().setCallback(new MqttCallback() {
      @Override
      public void connectionLost(Throwable cause) {}

      @Override
      public void messageArrived(String topic, MqttMessage message) {
        String messageStr = new String(message.getPayload());
        getCallback().onMqttMessageReceived(messageStr);
      }

      @Override
      public void deliveryComplete(IMqttDeliveryToken token) {}
    });
    getMqttConnectOptions().setAutomaticReconnect(true);
    getMqttConnectOptions().setCleanSession(false);
  }

  public void connect() {
    try {
      getMqttClient().connect(getMqttConnectOptions(), null, new IMqttActionListener() {
        @Override
        public void onSuccess(IMqttToken asyncActionToken) {
          getCallback().onMqttConnect();
        }

        @Override
        public void onFailure(IMqttToken asyncActionToken, Throwable exception) {
          Log.e(TAG, "error during connecting", exception);
        }

      });
    } catch (MqttException e) {
      Log.e(TAG, "error trying to connect", e);
    }
  }

  public void subscribe(String topic) {
    try {
      getMqttClient().subscribe(topic, MQTT_QOS);
    } catch (MqttException e) {
      Log.e(TAG, "error when subscribing", e);
    }
  }

  public void unsubscribe(String topic) {
    try {
      getMqttClient().unsubscribe(topic);
    } catch (MqttException e) {
      Log.e(TAG, "error when unsubscribe", e);
    }
  }

  public void disconnect() {
    try {
      getMqttClient().disconnect();
    } catch (MqttException e) {
      Log.e(TAG, "error when disconnect", e);
    }
  }

  private MqttAndroidClient getMqttClient() {
    return this.mqttClient;
  }

  private MqttConnectOptions getMqttConnectOptions() {
    return this.mqttConnectOptions;
  }

  private String getMqttServerUri() {
    return this.mqttServerUri;
  }

  private String getClientId() {
    return this.clientId;
  }

  private IMqttListener getCallback() {
    return this.callback;
  }

}
