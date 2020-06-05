package cm.in.tum.de.localvlc;

import android.content.Context;
import android.net.wifi.WifiConfiguration;
import android.net.wifi.WifiManager;
import android.widget.Toast;

public class WifiNetworks {

  private final GUI gui;
  private final Context context;
  private final WifiManager wifiManager;
  private String targetSsid;

  public WifiNetworks(Context context, GUI gui) {
    this.gui = gui;
    this.context = context;
    this.wifiManager = (WifiManager) getContext()
            .getApplicationContext().getSystemService(Context.WIFI_SERVICE);
    if (!getWifiManager().isWifiEnabled()) {
      getWifiManager().setWifiEnabled(true);
    }
  }

  public void connect(String ssid, String password) {
    setTargetSsid(String.format("\"%s\"", ssid));
    if (isNetworkExisting(getTargetSsid())) {
      adaptExistingNetwork(password);
    } else {
      addNewNetwork(getTargetSsid(), password);
    }
  }

  private void adaptExistingNetwork(String password) {
    for( WifiConfiguration wifiConfig : getWifiManager().getConfiguredNetworks()) {
      if (wifiConfig.SSID.equals(getTargetSsid())) {
        wifiConfig.preSharedKey = String.format("\"%s\"", password);
        int netId = getWifiManager().updateNetwork(wifiConfig);
        if (netId != -1) {
          getWifiManager().disconnect();
          getWifiManager().enableNetwork(wifiConfig.networkId, true);
          performGuiAction();
          break;
        } else {
          Toast.makeText(getContext(),
                  "Failed automatic Wi-Fi connection",
                  Toast.LENGTH_LONG).show();
        }
      }
    }
  }

  private void addNewNetwork(String ssid, String password) {
    WifiConfiguration wifiConfig = new WifiConfiguration();
    wifiConfig.SSID = ssid;
    wifiConfig.preSharedKey = String.format("\"%s\"", password);
    int netId = getWifiManager().addNetwork(wifiConfig);
    if (netId != -1) {
      getWifiManager().disconnect();
      getWifiManager().enableNetwork(netId, true);
      performGuiAction();
    } else {
      Toast.makeText(getContext(),
              "Failed automatic Wi-Fi connection",
              Toast.LENGTH_LONG).show();
    }
  }

  private void performGuiAction() {
    Toast.makeText(getContext(),
            "Successful automatic Wi-Fi authentication",
            Toast.LENGTH_SHORT).show();
    getGui().setWifiName(getTargetSsid());
  }

  private boolean isNetworkExisting(String ssid) {
    for( WifiConfiguration wifiConfig : getWifiManager().getConfiguredNetworks()) {
      if (wifiConfig.SSID.equals(ssid)) {
        return true;
      }
    }
    return false;
  }

  private WifiManager getWifiManager() {
    return this.wifiManager;
  }

  private Context getContext() {
    return this.context;
  }

  private GUI getGui() {
    return this.gui;
  }

  private void setTargetSsid(String ssid) {
    this.targetSsid = ssid;
  }

  private String getTargetSsid() {
    return this.targetSsid;
  }

}
