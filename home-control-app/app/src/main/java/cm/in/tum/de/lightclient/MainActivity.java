package cm.in.tum.de.lightclient;

import android.app.Activity;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.webkit.WebView;

import cm.in.tum.de.lightclient.home_control.CustomWebViewClient;
import cm.in.tum.de.lightclient.rest.RestfulService;
import cm.in.tum.de.lightclient.utils.AppStorage;
import cm.in.tum.de.lightclient.utils.Config;
import cm.in.tum.de.lightclient.utils.ConfigSection;
import cm.in.tum.de.lightclient.utils.EvaluationData;
import cm.in.tum.de.lightclient.utils.GUIUtils;
import cm.in.tum.de.lightclient.utils.IPermissionListener;
import cm.in.tum.de.lightclient.utils.PermissionHandler;
import cm.in.tum.de.lightclient.utils.StopWatch;
import cm.in.tum.de.lightclient.wifi.IWifiListener;
import cm.in.tum.de.lightclient.wifi.WifiService;

public class MainActivity extends AppCompatActivity implements IPermissionListener, IWifiListener {

  private static final String TAG = MainActivity.class.getSimpleName();
  private static Activity ACTIVITY;
  private PermissionHandler permissionHandler;
  private WebView webView;
  private Config config;
  private WifiService wifiService;
  private RequestServer requestServer;
  private RestfulService restfulService;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    ACTIVITY = this;
    this.config = new Config(getConfigResource());
    this.webView = new WebView(this);
    setContentView(getWebView());
    getWebView().getSettings().setJavaScriptEnabled(true);
    getWebView().setWebViewClient(new CustomWebViewClient());
    this.permissionHandler = new PermissionHandler(this, this);
    getPermissionHandler().checkPermissions();
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
                                         String permissions[],
                                         int[] grantResults) {
    getPermissionHandler().onRequestPermissionsResult(requestCode, grantResults);
  }

  private void startHomeControl(String ipAddress) {
    String url = getConfigValue("Service Protocol", ConfigSection.HOME_CONTROL) + "://" +
            ipAddress + ":" + getConfigValue("Service Port", ConfigSection.HOME_CONTROL);
    getWebView().loadUrl(url);
  }

  @Override
  public void onWifiConnected(String ssid, String serviceIpAddress) {
    this.requestServer = new RequestServer();
    String ipHomeControl = getGateway(serviceIpAddress);
    this.restfulService = new RestfulService(getRequestServer(),
            ipHomeControl, getConfigResource());
    getRequestServer().setRestfulService(getRestfulService());
    //getRequestServer().start();
    //getRestfulService().getToken();
    //getRestfulService().getEncryptedData("123456");
    String mode = getConfigValue("Mode", ConfigSection.SETTINGS);
    if (mode.contains("evaluate")) {
      AppStorage.createDataFolder();
      EvaluationData evaluationData = new EvaluationData();
      StopWatch stopWatch = new StopWatch();
      int counter = 0;
      String rounds = getConfigValue("Evaluation Rounds", ConfigSection.SETTINGS);
      int rounds_int = Integer.valueOf(rounds);
      while (counter < rounds_int) {
        stopWatch.start();
        boolean result = getRestfulService().evaluateAuthenticaton();
        stopWatch.stop();
        evaluationData.addAuthenticationResult(stopWatch.getElapsedTime(), result);
        counter++;
      }
      evaluationData.saveAuthentication();
      /*try {
        getRequestServer().stop();
      } catch (IOException e) {
        Log.d(TAG, "close request server");
      } catch (InterruptedException e) {
        Log.d(TAG, "close request server");
      }*/
      String title = "Evaluation";
      String message = rounds + " evaluation rounds are completed.";
      GUIUtils.showOKAlertDialog(this, title, message);
    } else {
      startHomeControl(ipHomeControl);
    }
  }

  @Override
  public void onPermissionsGranted() {
    this.wifiService = new WifiService(this, this, getConfigResource());
    getWifiService().checkConnectivity();
  }

  @Override
  public void onPermissionsNotGranted() {
    String title = "Permission Handler";
    String message = "Not all required permissions are granted, please allow necessary permissions";
    GUIUtils.showOKAlertDialog(this, title, message);
  }

  private String getGateway(String serviceIpAddress) {
    return serviceIpAddress.substring(0, serviceIpAddress.lastIndexOf(".") + 1) + "1";
  }

  private WebView getWebView() {
    return this.webView;
  }

  private RestfulService getRestfulService() {
    return this.restfulService;
  }

  private PermissionHandler getPermissionHandler() {
    return this.permissionHandler;
  }

  private WifiService getWifiService() {
    return this.wifiService;
  }

  private RequestServer getRequestServer() {
    return this.requestServer;
  }

  private Config getConfig() {
    return this.config;
  }

  private String getConfigValue(String key, String section) {
    return getConfig().get(key, section, String.class);
  }

  public static Activity getInstance() {
    return ACTIVITY;
  }

  public static int getCertificateResource() {
    return R.raw.server;
  }

  public static int getConfigResource() {
    return R.raw.config;
  }

}
