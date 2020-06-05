package cm.in.tum.de.localvlc;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.FileReader;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.Enumeration;
import java.util.UUID;

public class MainActivity extends AppCompatActivity implements IMqttListener {

  private static final String TAG = MainActivity.class.getSimpleName();
  private static final String IP_USB_RANGE = "192.168.42.";
  private static final String USB_TETHERING_INTERFACE = "rndis";
  private static final String ACTION_TETHER_STATE_CHANGED = "android.net.conn.TETHER_STATE_CHANGED";

  private static final String MQTT_TOPIC = "wifi_authentication";
  private static final int SSID_IDX = 0;
  private static final int PASSWORD_IDX = 1;

  private GUI gui;
  private MqttClient mqttClient;
  private WifiNetworks wifiNetworks;
  private String lastPassword;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    this.gui = new GUI(this,
            findViewById(R.id.statusUsbTethering),
            findViewById(R.id.statusConnLightReceiver),
            findViewById(R.id.ipLightReceiver),
            findViewById(R.id.wifiName),
            findViewById(R.id.tableLoginData));
    // Test
    //this.wifiNetworks = new WifiNetworks(this, getGui());
    //getWifiNetworks().connect("LocalVLC", "12345693");
    if (!isUsbTetheringActive()) {
      IntentFilter filter = new IntentFilter(ACTION_TETHER_STATE_CHANGED);
      getApplicationContext().registerReceiver(new TetherChange(), filter);
      Intent intent = new Intent();
      intent.setClassName("com.android.settings",
              "com.android.settings.TetherSettings");
      startActivity(intent);
      Toast.makeText(this, "Please enable USB tethering",
              Toast.LENGTH_SHORT).show();
    } else {
      handleUsbTethering(getUsbThetheredIP());
    }
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (getMqttClient() != null) {
      getMqttClient().unsubscribe(MQTT_TOPIC);
      getMqttClient().disconnect();
    }
  }

  private void handleUsbTethering(String serverIp) {
    getGui().setStatusUsbTethering("active");
    getGui().setIpLightReceiver(serverIp);
    this.wifiNetworks = new WifiNetworks(this, getGui());
    String clientId = UUID.randomUUID().toString();
    if (serverIp != null && clientId != null) {
      this.mqttClient = new MqttClient(this, serverIp, clientId, this);
      getGui().setStatusConnLightReceiver("Try to connect ...");
      getMqttClient().connect();
    }
  }

  @Override
  public void onMqttConnect() {
    getGui().setStatusConnLightReceiver("connected");
    getMqttClient().subscribe(MQTT_TOPIC);
  }

  @Override
  public void onMqttMessageReceived(String message) {
    if (message.contains(":")) {
      String[] parts = message.split(":");
      String ssid = parts[SSID_IDX].trim();
      String password = parts[PASSWORD_IDX].trim();
      if (getLastPassword() != null) {
        if(!getLastPassword().equals(password)) {
          performWifiAction(ssid, password);
        }
      } else {
        performWifiAction(ssid, password);
      }
    } else {
      Toast.makeText(this,
              "Wrong Wi-Fi authentication message",
              Toast.LENGTH_SHORT).show();
    }
  }

  private void performWifiAction(String ssid, String password) {
    setLastPassword(password);
    getGui().addLoginData(ssid, password);
    getWifiNetworks().connect(ssid, password);
  }

  public class TetherChange extends BroadcastReceiver {
    @Override
    public void onReceive(final Context context, Intent intent) {
      if (isUsbTetheringActive()) {
        String serverIp = null;
        while (serverIp == null) {
          serverIp = getUsbThetheredIP();
          try {
            Thread.sleep(250);
          } catch (InterruptedException e) {
          }
        }
        // Show main activity
        Intent mainIntent = new Intent(context, MainActivity.class);
        mainIntent.setFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
        startActivity(mainIntent);
        handleUsbTethering(serverIp);
      }
    }
  }

  private boolean isUsbTetheringActive() {
    try {
      boolean usbIfaceFound = false;
      Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
      while (!usbIfaceFound && interfaces.hasMoreElements()) {
        NetworkInterface iface = interfaces.nextElement();
        if (iface.getName().contains(USB_TETHERING_INTERFACE)) {
          return true;
        }
      }
    } catch (SocketException e) {
      Log.d(TAG, "USB tethering active", e);
    }
    return false;
  }

  private String getUsbThetheredIP() {
    try {
      String line;
      BufferedReader bufferedReader = new BufferedReader(new FileReader("/proc/net/arp"));
      while ((line = bufferedReader.readLine()) != null) {
        String[] splitted = line.split(" +");
        if (splitted != null && splitted.length >= 4) {
          String ip = splitted[0];
          String mac = splitted[3];
          if (!mac.matches("00:00:00:00:00:00")
                  && ip.startsWith(IP_USB_RANGE)) {
            return ip;
          }
        }
      }
    } catch (Exception e) {
      Log.e(TAG, "IP of USB", e);
    }
    return null;
  }

  private WifiNetworks getWifiNetworks() {
    return this.wifiNetworks;
  }

  private MqttClient getMqttClient() {
    return this.mqttClient;
  }

  private GUI getGui() {
    return this.gui;
  }

  private void setLastPassword(String password) {
    this.lastPassword = password;
  }

  private String getLastPassword() {
    return this.lastPassword;
  }

}
