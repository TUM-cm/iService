import os
import sys
import pexpect
import netifaces
import subprocess
import user_interface_access_point
import utils.kernel_module as kernel_module
import utils.wifi_connector as wifi_connector

host_light_receiver = "192.168.0.1" # demo setting
#host_light_receiver = "131.159.24.69" # chair network
send_kernel_module = "send_light_kernel.ko"
port = 5000

if not os.geteuid() == 0:
    sys.exit("\nOnly root can run this script\n")

print("activate network adapter")
#wifi_connector.activate()
pexpect.run("ifconfig wlan0 up 10.0.0.1 netmask 255.255.255.0")
wifi_connector.activate_dhcp()

print("make kernel module")
directory_send_module = "send_light/"
if not os.path.isfile(directory_send_module + send_kernel_module):
    subprocess.call("make", shell=True, cwd=directory_send_module)

print("remove kernel module")
kernel_module.remove(send_kernel_module)

if not os.path.isfile("demo_wifi_login/hostapd_template.conf"):
    sys.exit("\nHostapd template is not available\n")
 
if not wifi_connector.interface_available():
    sys.exit("\nWifi interface is not available\n")

local_ip = netifaces.ifaddresses("eth0")[netifaces.AF_INET][0]['addr']
print("server GUI: " + local_ip + ":" + str(port))

print("start user interface")
user_interface_access_point.start(send_kernel_module, host_light_receiver)
