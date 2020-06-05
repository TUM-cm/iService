# WiFi
vi /etc/udev/rules.d/70-persistent-net.rules
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", ATTR{address}=="18:d6:c7:1d:49:eb", ATTR{dev_id}=="0x0", ATTR{type}=="1", KERNEL=="wlan*", NAME="wlan0"
/etc/init.d/networking restart

ifconfig wlan0 up 10.0.0.1 netmask 255.255.255.0

apt-get install dnsmasq
via  /etc/dnsmasq.conf 
interface=wlan0
dhcp-range=10.0.0.3,10.0.0.20,12h

killall wpa_supplicant
apt-get remove wpasupplicant

git clone https://github.com/lwfinger/rtl8188eu
cd rtl8188eu/hostapd-0.8/hostapd
cp defconfig .config
make
make install
cp /usr/local/bin/hostapd /usr/sbin

--------------------------------

# IoT Board
apt-get install libffi-dev
pip install --upgrade pip
pip install --upgrade setuptools
python -m easy_install --upgrade pyopenssl
	pip install --upgrade pyopenssl
pip install flask-httpauth
pip install netifaces
pip install flask
pip install twisted[tls]

# Windows
* Error:
from cryptography.hazmat.bindings._openssl import ffi, lib
ImportError: DLL load failed: %1 is not a valid Win32 application.
* Solution:
Install  Win64OpenSSL_Light-1_0_2 from https://slproweb.com/products/Win32OpenSSL.html
conda install -c anaconda cryptography

* Error:
from cryptography.hazmat.bindings._openssl import ffi, lib
ImportError: DLL load failed: Das angegebene Modul wurde nicht gefunden.
* Solution:
pip install --ignore-installed cryptography

--------------------------------

# Demo: Light bulb
vi start-light.sh
chmod +x start-light.sh
chmod 755 start-light.sh

#!/bin/bash
cd /sys/class/gpio
echo 49 > export
cd gpio49
echo out > direction
echo 1 > value

vi /etc/rc.local
/home/openvlc/start-light.sh

vi stop-light.sh
chmod +x stop-light.sh

#!/bin/bash
cd /sys/class/gpio/gpio49
echo out > direction
echo 0 > value
cd ..
echo 49 > unexport

--------------------------------

# Light bulb
* 3D mododel from Autodesk
* Print chain
1. Autodesk > stl (model)
2. stl > Prusa (printer) > gcode
3. gcode > OctoPrint

--------------------------------

# Packages for online simulator
sudo apt-get install r-base 
sudo pip install rpy2==2.8.6 (latet with python2 support)
sudo pip install dtaidistance==1.1.3
replace util.py for python2.7 (runs actually only with python3)
sudo pip install mock
sudo apt-get install python-dev libffi-dev libssl-dev
sudo pip install twisted[tls]
sudo R
install.packages("/home/haus/coupling_simulator/src/coupling/R-packages/proxy_0.4-16.tar.gz", repos = NULL, type="source")
install.packages("/home/haus/coupling_simulator/src/coupling/R-packages/dtw_1.18-1.tar.gz", repos = NULL, type="source")
sudo pip install pexpect

download from https://snap.stanford.edu/snappy/release/
Windows
	unzip
Linux
	tar xf

cd snap-...
sudo python2 setup.py install

Install C++ compiler for python27
No stdint.h, copy it to C:\Users\iostream\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\VC\include
https://raw.githubusercontent.com/mattn/gntp-send/master/include/msinttypes/stdint.h

sudo pip uninstall cvxpy
sudo pip install cvxpy==0.4

sudo pip install -U scikit-learn
sudo pip install -U numpy
sudo pip install -U scipy

newer R on Ubuntu 16 (otherwise proxy packet missing):

sudo echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -
sudo apt-get update
sudo apt-get install r-base r-base-dev

apt-get install python3-pip
pip3 install pandas
pip3 install scipy
pip3 install matplotlib
apt-get install python3-tk
pip3 install sklearn
apt-get install libffi-dev
apt-get install python3.7-dev
sudo pip3 install --upgrade --force-reinstall setuptools
pip3 install rpy2

hash -d pip3 (ImportError: cannot import name 'main')

--------------------------------

problem using threads and tsfresh
tsfresh is using not thread-safe https://pypi.org/project/tqdm/ for progress bar
latest version does not fix error (https://github.com/tqdm/tqdm/issues/510, https://github.com/tqdm/tqdm/issues/323)
solution: disable progress bar (disable_progressbar=True)
https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/extraction.html

--------------------------------

# install tsfresh under Linux with compatibility issues
sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo apt-get install atlas-devel
pip install --ugprade scipy
pip install --upgrade numpy
sudo apt-get purge python-numpy
sudo apt-get install python-numpy
pip install tsfresh

--------------------------------

sudo add-apt-repository ppa:jonathonf/python-2.7
sudo apt-get update
sudo apt-get install python2.7

--------------------------------

# install tsfresh on BBB: https://beagleboard.org/latest-images
Install alpha image with Debian buster otherwise operating system is too outdated
In summary: download image from BBB website, use Etcher to flash image on SD card,
boot from SD card, change /boot/uEnv.txt, enable commented out flasher, reboot
Next steps:

apt-get update
apt-get install python-pip
pip install tsfresh

screen -S <name>
detach: Ctrl + A + D
take: screen -r
list: screen -ls

deprecated:
apt-get install python-numpy
apt-get install python-scipy
apt-get install python-pandas (warning with 0.23.3)
apt-get install python-sklearn

update python35 to python37
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7
sudo easy_install3 pip

--------------------------------

download get-pip.py: https://pip.pypa.io/en/latest/installing/#install-or-upgrade-pip
open cmd as Administrator
python2 get-pip.py
cd C:\Users\iostream\Anaconda2\Scripts
pip.exe install tsfresh

--------------------------------

# Testbeds

server: ssh haus@emu03.cm.in.tum.de
VM (smc): ssh haus@172.24.25.25

smc test board
	ssh debian@vlc04.cm.in.tum.de
	pw: temppwd

coupling test board (with Debian buster for tsfresh)
	ssh debian@vlc02.cm.in.tum.de
	pw: temppwd
