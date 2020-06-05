all::
	g++ -o helib helib.cpp fhe.a -pthread -lntl -lgmp
	
test::
	g++ -g -O2 -std=c++11 -pthread -DFHE_THREADS -DFHE_BOOT_THREADS -o helib helib.cpp fhe.a -L/usr/local/lib -lntl -lgmp -lm

clean::
	$(RM) -R helib
