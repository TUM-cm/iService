# Secure multiparty computation
Overview homomorphic encryption: https://blog.quarkslab.com/a-brief-survey-of-fully-homomorphic-encryption-computing-on-encrypted-data.html
Secure computation with HElib: http://tommd.github.io/posts/HELib-Intro.html
Maturity and Performance of Programmable Secure Computation: https://eprint.iacr.org/2015/1039.pdf
JavaScript secure multiparty computation: https://github.com/multiparty/jiff

# Testbed SMC
apt-get update
apt-get install m4
apt-get install build-essential # gcc, make, etc.
(apt-get install gcc)

Install the gcc-7 packages:
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo add-apt-repository --remove ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y
Set it up so the symbolic links gcc, g++ point to the newer version:
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7 
sudo update-alternatives --config gcc
gcc --version
g++ --version

sudo vi /etc/apt/sources.list
deb http://ftp.us.debian.org/debian testing main contrib non-free
sudo apt-get update
sudo apt-get install -t testing g++
sudo apt-get install -t testing gcc
sudo apt-get install -t testing build-essential

# HElib
apt install git
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.lz
apt-get install lzip
tar --lzip -xf gmp-6.1.2.tar.lz
cd gmp-6.1.2
./configure (if error apt-get install m4)
sudo make
sudo make install

Fallback: wget https://www.shoup.net/ntl/ntl-10.5.0.tar.gz
wget https://www.shoup.net/ntl/ntl-11.2.1.tar.gz
wget -e use_proxy=yes -e http_proxy=$proxy http://www.shoup.net/ntl/ntl-11.2.1.tar.gz
tar xf ntl*
cd ntl-XXX/src
sudo ./configure NTL_GMP_LIP=on
sudo make
sudo make install

git clone http://github.com/shaih/HElib.git
cd HElib/src
sudo make
sudo make check

# SEAL
(apt install cmake)
sudo apt remove cmake
sudo apt purge --auto-remove cmake
wget https://cmake.org/files/v3.12/cmake-3.12.0-rc1.tar.gz
wget https://cmake.org/files/v3.12/cmake-3.12.1.tar.gz
tar xf cmake*
cd cmake*
./bootstrap
sudo make
sudo make install
cmake --version

https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/
tar xf SEAL*
cd SEAL*

- SEAL (main library)
cd SEAL
sudo cmake .
sudo make
sudo make install

- SEALExamples (extensively documented examples of using SEAL)
cd SEALExamples
sudo cmake .
sudo make

Error: Target "sealexamples" links to target "Threads::Threads" but the target was not found
Cause: missing find package for THREADS package

Edit SEAL_2.3.1/SEALExamples/CMakeLists.txt
	project(SEALExamples VERSION 2.3.1 LANGUAGES CXX)
	find_package(Threads REQUIRED)

Tests: SEAL_2.3.1/bin/sealexamples

Performance with Threads
	Mainly about memory sharing and mutex
	Homomorphic functions don't use parallelization at all

no multi-threading inside functions
perform a large number of allocations from a thread-safe memory pool
	slow when several threads are used

create local memory pools either thread-safe (slower) or thread-unsafe (faster)
Again, the performance difference might only show up when a large number of threads are used

# ABY
sudo apt-get update
apt-get install g++
apt-get install make
apt-get install cmake
sudo apt-get install libgmp-dev
sudo apt-get install libssl-dev

older version: git checkout <commit identifier>

old:
git clone --recursive https://github.com/encryptogroup/ABY.git
cd ABY/
make

new:
git clone https://github.com/encryptogroup/ABY.git
cd ABY/
mkdir build && cd build
cmake ..
cmake .. -DABY_BUILD_EXE=On (to build ABY test and benchmark executables)
cmake .. -DCMAKE_BUILD_TYPE=Release (enable optimizations)
cmake .. -DCMAKE_BUILD_TYPE=Debug (include debug symbols)

cmake .. -DCMAKE_BUILD_TYPE=Release -DABY_BUILD_EXE=On

CXX=/usr/bin/clang++ cmake .. (different compiler)
make
sudo make install

change #include <filesystem> to #include <experimental/filesystem>
std::filesystem to std::experimental::filesystem

* ABY at IoT Board
Flash BeagleBone with latest Debian
	Download image
	Etcher to flash image to SD card
	Boot from SD card and change data
		sudo vi /boot/uEnv.txt 
		turn images into eMMC flasher images
		edit the /boot/uEnv.txt file on the Linux partition on the microSD card
		remove the '#' on the line with 'cmdline=init=/opt/scripts/tools/eMMC/init-eMMC-flasher-v3.sh'
	Reboot to flash eMMC
	Login
		user: debian
		passwd: temppwd

Install docker for Debian
	docker load -i test
	docker image ls
	docker run test

Update kernel
	apt-cache search linux-image
	apt-get install <image>
	reboot
	sudo apt-get install linux-headers-$(uname -r)

# PPT
cmake . -DENABLE_SEAL=On -DENABLE_HELIB=On -DENABLE_ABY=Off
make
./ppt <testbed>

# Pyfhel
Python API for SEAL, HElib, Palisade
https://github.com/ibarrond/Pyfhel
