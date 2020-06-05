package cm.in.tum.de.localvlc;

import android.content.Context;
import android.view.View;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

public class GUI {

  private final Context context;
  private final TextView statusUsbTethering;
  private final TextView statusConnLightReceiver;
  private final TextView ipLightReceiver;
  private final TextView wifiName;
  private final TableLayout table;

  public GUI(Context context,
             View statusUsbTethering,
             View statusConnLightReceiver,
             View ipLightReceiver,
             View wifiName,
             View table) {
    this.context = context;
    this.statusUsbTethering = (TextView) statusUsbTethering;
    this.statusConnLightReceiver = (TextView) statusConnLightReceiver;
    this.ipLightReceiver = (TextView) ipLightReceiver;
    this.wifiName = (TextView) wifiName;
    this.table = (TableLayout) table;
  }

  public void setStatusUsbTethering(String status) {
    this.statusUsbTethering.setText(status);
  }

  public void setStatusConnLightReceiver(String status) {
    this.statusConnLightReceiver.setText(status);
  }

  public void setIpLightReceiver(String ip) {
    this.ipLightReceiver.setText(ip);
  }

  public void setWifiName(String wifiName) {
    this.wifiName.setText(wifiName);
  }

  public void addLoginData(String ssid, String password) {
    TableRow row = new TableRow(getContext());
    TextView value = new TextView(getContext());
    value.setText(getTableContent(ssid, password));
    row.addView(value);
    getTable().addView(row,
            new TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT,
                    TableLayout.LayoutParams.WRAP_CONTENT));
  }

  private String getTableContent(String ssid, String password) {
    return String.format("SSID: %10s                         Password: %10s", ssid, password);
  }

  private TableLayout getTable() {
    return this.table;
  }

  private Context getContext() {
    return this.context;
  }

}
