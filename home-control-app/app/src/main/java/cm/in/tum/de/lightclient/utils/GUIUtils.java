package cm.in.tum.de.lightclient.utils;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;

public class GUIUtils {

  private static AlertDialog.Builder createFinishAlertDialog(final Activity activity,
                                                             String title, String message) {
    DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
      @Override
      public void onClick(DialogInterface dialog, int which) {
        switch (which) {
          case DialogInterface.BUTTON_POSITIVE:
            activity.finish();
            break;
        }
      }
    };
    final AlertDialog.Builder dialog = new AlertDialog.Builder(activity);
    dialog.setTitle(title);
    dialog.setMessage(message);
    dialog.setPositiveButton("OK", dialogClickListener);
    return dialog;
  }

  public static void showOKAlertDialog(final Activity activity, String title, String message) {
    final AlertDialog.Builder dialog = createFinishAlertDialog(activity, title, message);
    activity.runOnUiThread(new Runnable() {
      @Override
      public void run() {
        dialog.show();
      }
    });
  }

}
