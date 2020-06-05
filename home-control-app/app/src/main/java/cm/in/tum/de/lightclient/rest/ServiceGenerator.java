package cm.in.tum.de.lightclient.rest;

import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;

import cm.in.tum.de.lightclient.MainActivity;
import cm.in.tum.de.lightclient.utils.FileUtils;
import okhttp3.OkHttpClient;
import retrofit2.Converter;
import retrofit2.Retrofit;

public class ServiceGenerator {

  public static final String TAG = ServiceGenerator.class.getSimpleName();
  private static final OkHttpClient.Builder httpClient = new OkHttpClient.Builder();

  public static Retrofit.Builder createBuilder(String baseUrl, Converter.Factory converter) {
    Retrofit.Builder builder = new Retrofit.Builder().baseUrl(baseUrl);
    builder.addConverterFactory(converter);
    return builder;
  }

  public static <S> S createService(Class<S> serviceClass, Retrofit.Builder builder, boolean dev) {
    try {
      SSLParameter sslParameter = getSSLConfig();
      httpClient.sslSocketFactory(sslParameter.getSslContext().getSocketFactory(),
              sslParameter.getTrustManager());
    } catch (CertificateException e) {
      Log.d(TAG, "CertificateException", e);
    } catch (IOException e) {
      Log.d(TAG, "IOException", e);
    } catch (KeyStoreException e) {
      Log.d(TAG, "KeyStoreException", e);
    } catch (NoSuchAlgorithmException e) {
      Log.d(TAG, "NoSuchAlgorithmException", e);
    } catch (KeyManagementException e) {
      Log.d(TAG, "KeyManagementException", e);
    }
    httpClient.connectTimeout(2, TimeUnit.MINUTES);
    httpClient.readTimeout(2, TimeUnit.MINUTES);
    if (dev) {
      // Workaround allow different dev environment without changing certificates
      httpClient.hostnameVerifier(new HostnameVerifier() {
        @Override
        public boolean verify(String hostname, SSLSession session) {
          return true;
        }
      });
    }
    OkHttpClient client = httpClient.build();
    Retrofit retrofit = builder.client(client).build();
    return retrofit.create(serviceClass);
  }

  private static SSLParameter getSSLConfig() throws CertificateException, IOException,
          KeyStoreException, NoSuchAlgorithmException, KeyManagementException {
    CertificateFactory cf = CertificateFactory.getInstance("X.509");
    try (InputStream cert = FileUtils.getInputStream(MainActivity.getCertificateResource())) {
      Certificate ca = cf.generateCertificate(cert);
      // Creating a KeyStore containing our trusted CAs
      String keyStoreType = KeyStore.getDefaultType();
      KeyStore keyStore = KeyStore.getInstance(keyStoreType);
      keyStore.load(null, null);
      keyStore.setCertificateEntry("ca", ca);
      // Creating a TrustManager that trusts the CAs in our KeyStore.
      String tmfAlgorithm = TrustManagerFactory.getDefaultAlgorithm();
      TrustManagerFactory tmf = TrustManagerFactory.getInstance(tmfAlgorithm);
      tmf.init(keyStore);
      // Creating an SSLSocketFactory that uses our TrustManager
      SSLContext sslContext = SSLContext.getInstance("TLS");
      sslContext.init(null, tmf.getTrustManagers(), null);
      TrustManager[] trustManagers = tmf.getTrustManagers();
      X509TrustManager trustManager = (X509TrustManager) trustManagers[0];
      return new SSLParameter(sslContext, trustManager);
    }
  }

  private static class SSLParameter {

    private final SSLContext sslContext;
    private final X509TrustManager trustManager;

    public SSLParameter(SSLContext sslContext, X509TrustManager trustManager) {
      this.sslContext = sslContext;
      this.trustManager = trustManager;
    }

    public SSLContext getSslContext() {
      return this.sslContext;
    }

    public X509TrustManager getTrustManager() {
      return this.trustManager;
    }

  }

}
