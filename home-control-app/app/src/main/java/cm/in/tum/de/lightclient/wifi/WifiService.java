package cm.in.tum.de.lightclient.wifi;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiConfiguration;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.SystemClock;
import android.util.Log;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import cm.in.tum.de.lightclient.utils.Config;
import cm.in.tum.de.lightclient.utils.ConfigSection;

public class WifiService extends BroadcastReceiver {

  private static final String TAG = WifiService.class.getSimpleName();
  private static final int WIFI_CONNECT_DURATION = 500;

  private final WifiManager wifiManager;
  private final Context applicationContext;
  private final IWifiListener wifiListener;
  private final Config config;
  private final String wifiPassword;
  private final String targetAP;

  public WifiService(Context applicationContext, IWifiListener wifiListener, int configResource) {
    this.config = new Config(configResource);
    this.applicationContext = applicationContext;
    this.wifiListener = wifiListener;
    this.wifiManager = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
    this.targetAP = getConfigValue("Target AP");
    this.wifiPassword = getConfigValue("Wifi Password");
  }

  public void checkConnectivity() {
    if (!getWifiManager().isWifiEnabled()) {
      getWifiManager().setWifiEnabled(true);
    }
    if (isConnected()) {
      callback();
    } else {
      getApplicationContext().registerReceiver(this,
              new IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION));
      getWifiManager().startScan();
    }
  }

  @Override
  public void onReceive(Context context, Intent intent) {
    List<ScanResult> scanResultList = getWifiManager().getScanResults();
    List<Wifi> wifis = new ArrayList<>();
    for(ScanResult scanResult : scanResultList) {
      if (scanResult.SSID.contains(getTargetAP())) {
        WifiAccess wifiAccess;
        if (scanResult.capabilities.contains("WPA")) {
          wifiAccess = new WifiAccess(WifiSecurity.WPA, getWifiPassword());
        } else if (scanResult.capabilities.contains("WEP")) {
          wifiAccess = new WifiAccess(WifiSecurity.WEP, getWifiPassword());
        } else {
          wifiAccess = new WifiAccess(WifiSecurity.Open, null);
        }
        wifis.add(new Wifi(scanResult.SSID, wifiAccess, scanResult.level));
      }
    }
    if (wifis.size() >= 1) {
      Collections.sort(wifis, new Comparator<Wifi>() {
        @Override
        public int compare(Wifi wifi1, Wifi wifi2) {
          return -Integer.compare(wifi1.getSignalLevel(), wifi2.getSignalLevel());
        }
      });
      // Avoid automatic reassociation with other networks
      for(WifiConfiguration configuredNetwork : getWifiManager().getConfiguredNetworks()) {
        getWifiManager().disableNetwork(configuredNetwork.networkId);
        getWifiManager().saveConfiguration();
      }
      boolean connected = false;
      for(Wifi wifi : wifis) {
        if (connected) {
          break;
        }
        removeWifi(wifi.getSsid());
        addWifi(wifi);
        // Connect
        for (WifiConfiguration configuredNetwork : getWifiManager().getConfiguredNetworks()) {
          if (configuredNetwork.SSID != null && configuredNetwork.SSID.equals(wifi.getSsid())) {
            connectWifi(configuredNetwork);
            SystemClock.sleep(WIFI_CONNECT_DURATION);
            connected = isConnected();
            break;
          }
        }
      }
      if (connected) {
        getApplicationContext().unregisterReceiver(this);
        callback();
      }
    }
  }

  private String convertIpAddress(int ipAddress) {
    try {
      final ByteBuffer byteBuffer = ByteBuffer.allocate(4);
      byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
      byteBuffer.putInt(ipAddress);
      final InetAddress inetAddress = InetAddress.getByAddress(null, byteBuffer.array());
      return inetAddress.getHostAddress();
    } catch (UnknownHostException e) {
      Log.d(TAG, "convert ip address");
    }
    return null;
  }

  private void removeWifi(String ssid) {
    for (WifiConfiguration configuredNetwork : getWifiManager().getConfiguredNetworks()) {
      if (configuredNetwork.SSID != null && configuredNetwork.SSID.equals(ssid)) {
        getWifiManager().removeNetwork(configuredNetwork.networkId);
        getWifiManager().saveConfiguration();
      }
    }
  }

  private void addWifi(Wifi wifi) {
    WifiConfiguration wifiConfiguration = createWifiConfiguration(wifi.getSsid(),
            wifi.getWifiAccess());
    getWifiManager().addNetwork(wifiConfiguration);
  }

  private boolean isConnected() {
    return getWifiManager().getConnectionInfo().getSSID().contains(getTargetAP());
  }

  private void connectWifi(WifiConfiguration wifiConfiguration) {
    getWifiManager().disconnect();
    getWifiManager().enableNetwork(wifiConfiguration.networkId, true);
  }

  private WifiConfiguration createWifiConfiguration(String ssid, WifiAccess wifiAccess) {
    WifiConfiguration wifiConfiguration = new WifiConfiguration();
    wifiConfiguration.SSID = ssid;
    switch(wifiAccess.getWifiSecurity()) {
      case WEP:
        wifiConfiguration.wepKeys[0] = "\"" + wifiAccess.getPassword() + "\"";
        wifiConfiguration.wepTxKeyIndex = 0;
        wifiConfiguration.allowedKeyManagement.set(WifiConfiguration.KeyMgmt.NONE);
        wifiConfiguration.allowedGroupCiphers.set(WifiConfiguration.GroupCipher.WEP40);
        break;
      case WPA:
        wifiConfiguration.preSharedKey = "\"" + wifiAccess.getPassword() + "\"";
        break;
      case Open:
        wifiConfiguration.allowedKeyManagement.set(WifiConfiguration.KeyMgmt.NONE);
        break;
    }
    return wifiConfiguration;
  }

  private void callback() {
    WifiInfo connInfo = getWifiManager().getConnectionInfo();
    String ipAddress = convertIpAddress(connInfo.getIpAddress());
    getWifiListener().onWifiConnected(connInfo.getSSID(), ipAddress);
  }

  private class Wifi {

    private final String ssid;
    private final WifiAccess wifiAccess;
    private final int signalLevel;

    public Wifi(String ssid, WifiAccess wifiAccess, int signalLevel) {
      this.ssid = String.format("\"%s\"", ssid);
      this.wifiAccess = wifiAccess;
      this.signalLevel = signalLevel;
    }

    public int getSignalLevel() {
      return this.signalLevel;
    }

    public String getSsid() {
      return this.ssid;
    }

    public WifiAccess getWifiAccess() {
      return this.wifiAccess;
    }

  }

  private class WifiAccess {

    private final WifiSecurity wifiSecurity;
    private final String password;

    public WifiAccess(WifiSecurity wifiSecurity, String password) {
      this.wifiSecurity = wifiSecurity;
      this.password = password;
    }

    public WifiSecurity getWifiSecurity() {
      return this.wifiSecurity;
    }

    public String getPassword() {
      return this.password;
    }

  }

  private enum WifiSecurity {
    WEP, WPA, Open
  }

  private WifiManager getWifiManager() {
    return this.wifiManager;
  }

  private Context getApplicationContext() {
    return this.applicationContext;
  }

  private IWifiListener getWifiListener() {
    return this.wifiListener;
  }

  private Config getConfig() {
    return this.config;
  }

  private String getTargetAP() {
    return this.targetAP;
  }

  private String getWifiPassword() {
    return this.wifiPassword;
  }

  private String getConfigValue(String key) {
    return getConfig().get(key, ConfigSection.LIGHT_RECEIVER, String.class);
  }

}
