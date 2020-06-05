package cm.in.tum.de.lightclient.utils;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

import java.util.ArrayList;
import java.util.List;

public class PermissionHandler {

  private static final int PERMISSION_MULTIPLE_REQUEST = 1;
  private static final String[] permissions = new String[] {
          Manifest.permission.ACCESS_COARSE_LOCATION,
          Manifest.permission.WRITE_EXTERNAL_STORAGE
  };

  private final Context context;
  private final IPermissionListener permissionListener;

  public PermissionHandler(Context context,
                           IPermissionListener permissionListener) {
    this.context = context;
    this.permissionListener = permissionListener;
  }

  public void checkPermissions() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      String[] permissionsToRequest = getPermissionsToRequest(getPermissions());
      if (permissionsToRequest.length > 0) {
        ActivityCompat.requestPermissions((Activity) getContext(),
                permissionsToRequest, PERMISSION_MULTIPLE_REQUEST);
      } else {
        getPermissionListener().onPermissionsGranted();
      }
    } else {
      getPermissionListener().onPermissionsGranted();
    }
  }

  private String[] getPermissionsToRequest(String[] requiredPermissions) {
    List<String> requestedPermissions = new ArrayList<>();
    for (String requiredPermission : requiredPermissions) {
      if (ContextCompat.checkSelfPermission(getContext(), requiredPermission)
              != PackageManager.PERMISSION_GRANTED) {
        requestedPermissions.add(requiredPermission);
      }
    }
    return requestedPermissions.toArray(new String[requestedPermissions.size()]);
  }

  public void onRequestPermissionsResult(int requestCode, int[] grantResults) {
    switch (requestCode) {
      case PERMISSION_MULTIPLE_REQUEST: {
        if (grantResults.length > 0) {
          boolean permissionsGranted = true;
          for(int grantResult : grantResults) {
            if (grantResult != PackageManager.PERMISSION_GRANTED) {
              permissionsGranted = false;
              break;
            }
          }
          if (permissionsGranted) {
            getPermissionListener().onPermissionsGranted();
          } else {
            getPermissionListener().onPermissionsNotGranted();
          }
        }
      }
    }
  }

  private String[] getPermissions() {
    return this.permissions;
  }

  private IPermissionListener getPermissionListener() {
    return this.permissionListener;
  }

  private Context getContext() {
    return this.context;
  }

}
