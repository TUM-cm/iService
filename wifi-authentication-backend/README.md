# Client-Server

## Server: python -m demo_wifi_login.control_access_point
killall wpa_supplicant
apt-get remove wpasupplicant

pip install paho-mqtt
pip install netifaces
pip install pexpect
pip install Flask-WTF
pip install flask-socketio
pip install pyotp
pip install wifi

### Improved Performance of Flask
pip install eventlet
or (seems to be faster)
pip install gevent
pip install gevent-websocket

apt-get update
apt-get install cython
apt-get install mosquitto mosquitto-clients

dpkg -i <.deb> # install
dpkg -r <package> # remove

vi /etc/udev/rules.d/70-persistent-net.rules
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", ATTR{address}=="74:da:38:ea:05:ab", ATTR{dev_id}=="0x0", ATTR{type}=="1", KERNEL=="wlan*", NAME="wlan0"
sudo reboot

apt-get install dnsmasq
via  /etc/dnsmasq.conf 
interface=wlan0
dhcp-range=10.0.0.3,10.0.0.20,12h

git clone https://github.com/lwfinger/rtl8188eu
cd /rtl8188eu/hostapd-0.8/hostapd
cp defconfig .config
make
make install
cp /usr/local/bin/hostapd /usr/sbin

## Client: python -m demo_wifi_login.light_client

pip install --upgrade setuptools
pip install netifaces
pip install paho-mqtt
apt-get udpate
apt-get install cython (slow: pip install Cython)
apt-get install mosquitto mosquitto-clients

# Setting

GUI (laptop): 192.168.0.3, 255.255.255.0, without default gateway

light receiver: 192.168.0.1
	sudo -s
	cd /src
	python -m demo_wifi_login.light_client

light sender: 192.168.0.2
	sudo -s
	cd /src
	python -m demo_wifi_login.control_access_point

# Performance measurements
pervasive
	Duration for data transmission		0.64 +/- 0.16 s
	Duration for Wi-Fi authentication	2.77 +/- 1.4 s

directional
	Duration for data transmission		0.63 +/- 0.22 s
	Duration for Wi-Fi authentication	1.93 +/- 1.33 s

# Network setting
connmanctl services
light receiver: connmanctl config ethernet_04a316afc86e_cable --ipv4 manual 192.168.0.1 255.255.255.0
light sender: connmanctl config ethernet_50658360ea1c_cable --ipv4 manual 192.168.0.2 255.255.255.0

# Network configuration
https://groups.google.com/forum/#!topic/beagleboard/yfNwIk_dWlg
connmanctl services
onnmanctl config <service> --ipv4 manual <ip_addr> 255.255.255.0
connmanctl config <service> --ipv4 dhcp

# Steps to start demo

1. Set static IP of laptop to show authentication monitor
	IP: 192.168.0.3
	Network: 255.255.255.0

2. Connect laptop and two Beaglebones to switch
	Use ethernet adapter for laptop (important!!!)
	
3. Connect smartphone to light receiver
	Start authentication app
	Start USB tethering
		
4. Start server
	cd src/
	python -m demo_wifi_login.control_access_point

5. Start receiver:
	cd src/
	python -m demo_wifi_login.light_client

6. Open authentication monitor at laptop in web browser (check url at server)
	Select LED
	Start demo
