package cm.in.tum.de.lightclient.home_control;

import android.net.http.SslError;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.SslErrorHandler;

public class CustomWebViewClient extends WebViewClient {

  public void onReceivedSslError(WebView view, final SslErrorHandler handler, SslError error) {
    handler.proceed(); // Ignore errors
  }

}
