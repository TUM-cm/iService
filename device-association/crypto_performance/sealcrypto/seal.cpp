#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <random>
#include <limits>

#include "seal/seal.h"

using namespace std;
using namespace seal;

double euclideanDistance(vector<Ciphertext> v1, vector<Ciphertext> v2,
			Evaluator evaluator, Decryptor decryptor, IntegerEncoder encoder)
{
	for (int i = 0; i < v1.size(); i++) {	
		evaluator.sub(v1[i], v2[i]);
		evaluator.multiply(v1[i], v1[i]);
	}
	Ciphertext sum;
	evaluator.add_many(v1, sum);
	Plaintext plain_result;
	decryptor.decrypt(sum, plain_result);
	double result = sqrt(encoder.decode_int64(plain_result));
	return result;
}

int64_t innerProduct(vector<Ciphertext> v1, vector<Ciphertext> v2,
		Evaluator evaluator, Decryptor decryptor, IntegerEncoder encoder)
{
	for(int i = 0; i < v1.size(); i++) {
		evaluator.multiply(v1[i], v2[i]);
	}
	Ciphertext sum;
	evaluator.add_many(v1, sum);
	Plaintext plain_result;
	decryptor.decrypt(sum, plain_result);
	return encoder.decode_int64(plain_result);
}

double cosineDistance(vector<Ciphertext> v1, vector<Ciphertext> v2,
		Evaluator evaluator, Decryptor decryptor, IntegerEncoder encoder)
{
	int64_t scalar_v12 = innerProduct(v1, v2, evaluator, decryptor, encoder);
	int64_t scalar_v11 = innerProduct(v1, v1, evaluator, decryptor, encoder);
	int64_t scalar_v22 = innerProduct(v2, v2, evaluator, decryptor, encoder);
	double result = 1.0;
	result -= scalar_v12 / (sqrt(scalar_v11) * sqrt(scalar_v22));
	return result;
}

void test(vector<int> plain_v1, vector<int> plain_v2)
{
	ChooserEncoder chooser_encoder(3);
	ChooserEvaluator chooser_evaluator;
	ChooserPoly c_input(10, 1);

	ChooserPoly c_cubed_input = chooser_evaluator.exponentiate(c_input, 3, 15);
	ChooserPoly c_term1 = chooser_evaluator.multiply_plain(c_cubed_input, chooser_encoder.encode(42));
	ChooserPoly c_term2 = chooser_evaluator.multiply_plain(c_input, chooser_encoder.encode(27));
	ChooserPoly c_sum12 = chooser_evaluator.sub(c_term1, c_term2);
	ChooserPoly c_result = chooser_evaluator.add_plain(c_sum12, chooser_encoder.encode(1));

    	EncryptionParameters optimal_parms;
    	chooser_evaluator.select_parameters({ c_result }, 0, optimal_parms);
	
	SEALContext optimal_context(optimal_parms);
	KeyGenerator keygen(optimal_context);
	PublicKey public_key = keygen.public_key();
	SecretKey secret_key = keygen.secret_key();

	Encryptor encryptor(optimal_context, public_key);
	Evaluator evaluator(optimal_context);
	Decryptor decryptor(optimal_context, secret_key);
	IntegerEncoder encoder(optimal_context.plain_modulus(), 3);

	vector<Ciphertext> v1;
	vector<Ciphertext> v2;
	for (int i = 0; i < plain_v1.size(); i++) {
		v1.emplace_back(optimal_parms);
		v2.emplace_back(optimal_parms);
		encryptor.encrypt(encoder.encode(plain_v1[i]), v1[i]);
		encryptor.encrypt(encoder.encode(plain_v2[i]), v2[i]);
	}
	double euclidean = euclideanDistance(v1, v2, evaluator, decryptor, encoder);
	double cosine = cosineDistance(v1, v2, evaluator, decryptor, encoder);
	cout << "euclidean: " << euclidean << endl;
	cout << "cosine: " << cosine << endl;
}

int main()
{
	int num = 3, min = 1, max = 500;
	vector<int> v1;
	vector<int> v2;
	srand(time(NULL));
	for (int i = 0; i < num; i++) {
		int val1 = rand()%(max-min + 1) + min;
		int val2 = rand()%(max-min + 1) + min;
		v1.emplace_back(val1);
		v2.emplace_back(val2);
	}

	for(int i=0; i < num; i++) {
		cout << v1[i] << " ";	
	}
	cout << endl;
	for(int i=0; i < num; i++) {
		cout << v2[i] << " ";	
	}
	cout << endl;
	test(v1, v2);
	return 0;
}

