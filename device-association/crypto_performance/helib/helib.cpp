#include <NTL/ZZX.h>
#include <NTL/RR.h>
#include <sys/time.h>
#include "FHE.h"
#include "EncryptedArray.h"
using namespace std;

class Timer {
public:
	void start() {
		m_start = my_clock();
	}
	void stop() {
		m_stop = my_clock();
	}
	double elapsed_time() const {
		return m_stop - m_start;
	}

private:
	double m_start, m_stop;
	double my_clock() const {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + tv.tv_usec * 1e-6;
	}
};

NTL::RR scalar_product(vector<long> v1, vector<long> v2,
		FHEPubKey publicKey, FHESecKey secretKey) {
	vector<Ctxt> encV1;
	vector<Ctxt> encV2;
	for (int i = 0; i < v1.size(); i++) {
		Ctxt tempV1(publicKey);
		publicKey.Encrypt(tempV1, to_ZZX(v1[i]));
		encV1.push_back(tempV1);
		Ctxt tempV2(publicKey);
		publicKey.Encrypt(tempV2, to_ZZX(v2[i]));
		encV2.push_back(tempV2);
	}
	cout << "created ciphertexts" << endl;
	for (int i = 0; i < encV2.size(); i++) {
		encV1[i] *= encV2[i];
		//ZZX result;
		//secretKey.Decrypt(result, encV1[i]);
		//cout << result << " ";
	}
	//cout << endl;
	for (int i = 1; i < encV1.size(); i++) {
		encV1[0] += encV1[i];
		//ZZX result;
		//secretKey.Decrypt(result, encV1[0]);
		//cout << result << " ";
	}
	//cout << endl;
	ZZX result;
	secretKey.Decrypt(result, encV1[0]);
	vector<Ctxt>().swap(encV1);
	vector<Ctxt>().swap(encV2);
	return to_RR(result[0]);
}

NTL::RR euclidean_distance(vector<long> v1, vector<long> v2,
		FHESecKey secretKey, FHEPubKey publicKey) {
	vector<Ctxt> encV1;
	vector<Ctxt> encV2;
	for (int i = 0; i < v1.size(); i++) {
		Ctxt tempV1(publicKey);
		publicKey.Encrypt(tempV1, to_ZZX(v1[i]));
		encV1.push_back(tempV1);

		Ctxt tempV2(publicKey);
		publicKey.Encrypt(tempV2, to_ZZX(v2[i]));
		encV2.push_back(tempV2);
	}
	cout << "created ciphertexts" << endl;
	for (int i = 0; i < encV2.size(); i++) {
		encV1[i] -= encV2[i];
	}
	for (int i = 0; i < encV1.size(); i++) {
		encV1[i] *= encV1[i];
	}
	for (int i = 1; i < encV1.size(); i++) {
		encV1[0] += encV1[i];
	}
	ZZX result;
	secretKey.Decrypt(result, encV1[0]);
	vector<Ctxt>().swap(encV1);
	vector<Ctxt>().swap(encV2);
	return SqrRoot(to_RR(result[0]));
}

NTL::RR cosine_distance(vector<long> v1, vector<long> v2,
		FHESecKey secretKey, FHEPubKey publicKey) {
	RR scalar_v12 = scalar_product(v1, v2, publicKey, secretKey);
	RR scalar_v1 = scalar_product(v1, v1, publicKey, secretKey);
	RR scalar_v2 = scalar_product(v2, v2, publicKey, secretKey);
	/*ZZX scalar_v1;
	secretKey.Decrypt(scalar_v1, enc_scalar_v1);
	ZZX scalar_v2;
	secretKey.Decrypt(scalar_v2, enc_scalar_v2);
	ZZX scalar_v12;
	secretKey.Decrypt(scalar_v12, enc_scalar_v12);*/
	cout << "v12: " << scalar_v12 << endl;	
	cout << "v11: " << scalar_v1 << endl;
	cout << "v22: " << scalar_v2 << endl;
	return to_RR(1) - (scalar_v12 / (SqrRoot(scalar_v1) * SqrRoot(scalar_v2)));
}

/*
 *   p       plaintext base  [ default=2 ]
 *   r       lifting  [ default=1 ]
 *   d       degree of the field extension  [ default=1 ]
 *   c       number of columns in the key-switching matrices  [ default=2 ]
 *   k       security parameter  [ default=80 ]
 *   L       # of levels in the modulus chain  [ default=heuristic ]
 *   s       minimum number of slots  [ default=0 
 *   m       use specified value as modulus
 */
void test(vector<long> v1, vector<long> v2) {
	Timer timer;
	timer.start();
	long p = 44000069; // p=257 > mod 257 > 8 bit int (0-256) must be a prime number
	long r = 1;
	long L = 5; // 6
	long c = 2;
	long w = 64;
	long d = 1;
	long k = 80;
	long s = 0;
	long m = FindM(k, L, c, p, d, s, 0);

	FHEcontext context(m, p, r);
	buildModChain(context, L, c);

	FHESecKey secretKey(context);
	const FHEPubKey& publicKey = secretKey;
	secretKey.GenSecKey(w); // A Hamming-weight-w secret key
	addSome1DMatrices(secretKey); // compute key-switching matrices that we need, time consuming
	timer.stop();
	cout << "Initialization" << endl;
	cout << "Duration: " << timer.elapsed_time() << " s" << endl;

	RR result;
	timer.start();
	result = euclidean_distance(v1, v2, secretKey, publicKey);
	timer.stop();
	cout << "Euclidean: " << result << endl;
	cout << "Duration: " << timer.elapsed_time() << " s" << endl;

	timer.start();
	result = cosine_distance(v1, v2, secretKey, publicKey);
	timer.stop();
	cout << "Cosine: " << result << endl;
	cout << "Duration: " << timer.elapsed_time() << " s" << endl;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		cout << "Amount of elements or two input vectors are mandatory" << endl;
		exit(0);
	}
	vector<long> v1;
	vector<long> v2;
	/*cout << "Start initialization ..." << endl;
	ArgMapping amap;
	amap.arg("v1", v1, "input vector", NULL);
	amap.note("e.g., v1='[5 3 187]'");
	amap.arg("v2", v2, "input vector", NULL);
	amap.note("e.g., v2='[5 3 187]'");
	amap.parse(argc, argv);*/

	int num = atoi(argv[1]);
	int min = 1, max = 500;
	cout << "len: " << num << endl;
	srand(time(NULL));
	for(int i = 0; i < num; i++) {
		int val1 = rand()%(max-min + 1) + min;
		int val2 = rand()%(max-min + 1) + min;
		v1.emplace_back(val1);
		v2.emplace_back(val2);	
	}

	for(int i = 0; i < v1.size(); i++) {
		if (i > 0) {
			cout << ",";		
		}
		cout << v1[i];	
	}
	cout << endl << endl;
	for(int i = 0; i < v2.size(); i++) {
		if (i > 0) {
			cout << ",";		
		}
		cout << v2[i];	
	}
	cout << endl;

	test(v1, v2);
	return 0;
}

