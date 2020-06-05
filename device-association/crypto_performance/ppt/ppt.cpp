#include <iostream>
#include <mutex>
#include <random>
#include <fstream>

#ifdef ENABLE_ABY
#include <thread>
#include <ENCRYPTO_utils/crypto/crypto.h>
#include "abycore/sharing/sharing.h"
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/arithmeticcircuits.h"
#include "abycore/circuit/circuit.h"
#endif

#ifdef ENABLE_SEAL
#include "seal/seal.h"
using namespace seal;
#endif

#ifdef ENABLE_HELIB
#include <NTL/ZZX.h>
#include <NTL/RR.h>
#include "FHE.h"
#include "EncryptedArray.h"
#endif

using namespace std;

#ifdef ENABLE_SEAL
double sealEuclideanDistance(vector<uint64_t> plain_v1, vector<uint64_t> plain_v2,
			EncryptionParameters optimal_parms, Encryptor encryptor,
			Evaluator evaluator, Decryptor decryptor, IntegerEncoder encoder)
{
	vector<Ciphertext> v1;
	vector<Ciphertext> v2;
	for (int i = 0; i < plain_v1.size(); i++) {
		v1.emplace_back(optimal_parms);
		v2.emplace_back(optimal_parms);
		encryptor.encrypt(encoder.encode(plain_v1[i]), v1[i]);
		encryptor.encrypt(encoder.encode(plain_v2[i]), v2[i]);
	}
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

int64_t sealInnerProduct(vector<uint64_t> plain_v1, vector<uint64_t> plain_v2,
		EncryptionParameters optimal_parms, Encryptor encryptor,
		Evaluator evaluator, Decryptor decryptor, IntegerEncoder encoder)
{
	vector<Ciphertext> v1;
	vector<Ciphertext> v2;
	for (int i = 0; i < plain_v1.size(); i++) {
		v1.emplace_back(optimal_parms);
		v2.emplace_back(optimal_parms);
		encryptor.encrypt(encoder.encode(plain_v1[i]), v1[i]);
		encryptor.encrypt(encoder.encode(plain_v2[i]), v2[i]);
	}
	for(int i = 0; i < v1.size(); i++) {
		evaluator.multiply(v1[i], v2[i]);
	}
	Ciphertext sum;
	evaluator.add_many(v1, sum);
	Plaintext plain_result;
	decryptor.decrypt(sum, plain_result);
	return encoder.decode_int64(plain_result);
}

double sealCosineDistance(vector<uint64_t> plain_v1, vector<uint64_t> plain_v2,
		EncryptionParameters optimal_parms, Encryptor encryptor,
		Evaluator evaluator, Decryptor decryptor, IntegerEncoder encoder)
{
	int64_t scalar_v12 = sealInnerProduct(plain_v1, plain_v2, optimal_parms, encryptor, evaluator, decryptor, encoder);
	int64_t scalar_v11 = sealInnerProduct(plain_v1, plain_v1, optimal_parms, encryptor, evaluator, decryptor, encoder);
	int64_t scalar_v22 = sealInnerProduct(plain_v2, plain_v2, optimal_parms, encryptor, evaluator, decryptor, encoder);
	double result = 1.0;
	result -= scalar_v12 / (sqrt(scalar_v11) * sqrt(scalar_v22));
	return result;
}

tuple<chrono::microseconds, chrono::microseconds, chrono::microseconds> testSeal(
	vector<uint64_t> plain_v1, vector<uint64_t> plain_v2)
{
	chrono::high_resolution_clock::time_point timeStart, timeEnd;
	
	timeStart = chrono::high_resolution_clock::now();
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
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeInitialisation = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Initialisation: " << timeInitialisation.count() << " us" << endl;

	timeStart = chrono::high_resolution_clock::now();
	double euclidean = sealEuclideanDistance(plain_v1, plain_v2, optimal_parms, encryptor, evaluator, decryptor, encoder);
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeDiffEuclidean = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Euclidean: " << euclidean << ", " << timeDiffEuclidean.count() << " us" << endl;
	
	timeStart = chrono::high_resolution_clock::now();
	double cosine = sealCosineDistance(plain_v1, plain_v2, optimal_parms, encryptor, evaluator, decryptor, encoder);
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeDiffCosine = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Cosine: " << cosine << ", " << timeDiffCosine.count() << " us" << endl;

	return make_tuple(timeInitialisation, timeDiffEuclidean, timeDiffCosine);	
}
#endif

#ifdef ENABLE_ABY
share* abyInnerProductCircuit(share *s_x, share *s_y, uint32_t num, ArithmeticCircuit *ac) {
	s_x = ac->PutMULGate(s_x, s_y);
	s_x = ac->PutSplitterGate(s_x);
	for (uint32_t i = 1; i < num; i++) {
		s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	}
	s_x->set_bitlength(1);
	return s_x;
}

share* abyEuclideanDistance(share *s_x, share *s_y, uint32_t num, ArithmeticCircuit *ac) {
	s_x = ac->PutSUBGate(s_x, s_y);
	s_x = ac->PutMULGate(s_x, s_x);
	s_x = ac->PutSplitterGate(s_x);
	for (uint32_t i = 1; i < num; i++) {
		s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	}
	s_x->set_bitlength(1);
	return s_x;
}

void abyParty(vector<uint64_t> v1, vector<uint64_t> v2,
		e_role role, const string& address, uint16_t port, seclvl seclvl,
		uint32_t bitlen, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing) {
	chrono::high_resolution_clock::time_point timeStart, timeEnd;
	timeStart = chrono::high_resolution_clock::now();
	ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg);
	vector<Sharing*>& sharings = party->GetSharings();
	ArithmeticCircuit* circ = (ArithmeticCircuit*) sharings[sharing]->GetCircuitBuildRoutine();
	timeEnd = chrono::high_resolution_clock::now();

	share *s_x_vec, *s_y_vec, *s_out;
	share *s_out_xy, *s_out_xx, *s_out_yy;
	uint64_t *xvals = &v1[0];
	uint64_t *yvals = &v2[0];
	int num = v1.size();
	s_x_vec = circ->PutSIMDINGate(num, xvals, bitlen, SERVER);
	s_y_vec = circ->PutSIMDINGate(num, yvals, bitlen, CLIENT);

	timeStart = chrono::high_resolution_clock::now();
	s_out = abyEuclideanDistance(s_x_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out = circ->PutOUTGate(s_out, ALL);
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeDiffEuclidean = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Euclidean: " << timeDiffEuclidean.count() << " us" << endl;

	timeStart = chrono::high_resolution_clock::now();
	s_out_xy = abyInnerProductCircuit(s_x_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out_xy = circ->PutOUTGate(s_out_xy, ALL);
	s_out_xx = abyInnerProductCircuit(s_x_vec, s_x_vec, num, (ArithmeticCircuit*) circ);
	s_out_xx = circ->PutOUTGate(s_out_xx, ALL);
	s_out_yy = abyInnerProductCircuit(s_y_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out_yy = circ->PutOUTGate(s_out_yy, ALL);
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeDiffCosine = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Cosine: " << timeDiffCosine.count() << " us" << endl;

	party->ExecCircuit();

	double euclidean = sqrt(s_out->get_clear_value<uint64_t>());

	uint64_t out_xy = s_out_xy->get_clear_value<uint64_t>();
	uint64_t out_xx = s_out_xx->get_clear_value<uint64_t>();
	uint64_t out_yy = s_out_yy->get_clear_value<uint64_t>();
	double cosine = 1.0;
	cosine -= out_xy / (sqrt(out_xx) * sqrt(out_yy));

	// return value by async thread
	cout << "euclidean: " << euclidean << endl;
	cout << "cosine: " << cosine << endl;
	
	delete s_x_vec;
	delete s_y_vec;
	delete party;
}

void testAby(vector<uint64_t> v1, vector<uint64_t> v2)
{
	e_sharing sharing = S_ARITH;
	uint32_t bitlen = 64; // 16, 32, 64
	uint32_t secparam = 128;
	uint32_t nthreads = 1;
	srand (time(NULL));
	uint16_t port = rand() % 10000 + 1;
	string address = "127.0.0.1";
	e_mt_gen_alg mt_alg = MT_OT;
	seclvl seclvl = get_sec_lvl(secparam);
	thread server(abyParty, v1, v2, SERVER, address, port, seclvl, bitlen, nthreads, mt_alg, sharing);
	thread client(abyParty, v1, v2, CLIENT, address, port, seclvl, bitlen, nthreads, mt_alg, sharing);
	server.join();
	client.join();
}
#endif

#ifdef ENABLE_HELIB
NTL::RR helibScalarProduct(vector<uint64_t> v1, vector<uint64_t> v2, FHEPubKey publicKey, FHESecKey secretKey) {
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
	for (int i = 0; i < encV2.size(); i++) {
		encV1[i] *= encV2[i];
	}
	for (int i = 1; i < encV1.size(); i++) {
		encV1[0] += encV1[i];
	}
	ZZX result;
	secretKey.Decrypt(result, encV1[0]);
	vector<Ctxt>().swap(encV1);
	vector<Ctxt>().swap(encV2);
	return to_RR(result[0]);
}

NTL::RR helibEuclideanDistance(vector<uint64_t> v1, vector<uint64_t> v2, FHESecKey secretKey, FHEPubKey publicKey) {
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

NTL::RR helibCosineDistance(vector<uint64_t> v1, vector<uint64_t> v2, FHESecKey secretKey, FHEPubKey publicKey) {
	RR scalar_v12 = helibScalarProduct(v1, v2, publicKey, secretKey);
	RR scalar_v1 = helibScalarProduct(v1, v1, publicKey, secretKey);
	RR scalar_v2 = helibScalarProduct(v2, v2, publicKey, secretKey);
	return to_RR(1) - (scalar_v12 / (SqrRoot(scalar_v1) * SqrRoot(scalar_v2)));
}

tuple<chrono::microseconds, chrono::microseconds, chrono::microseconds> testHelib(vector<uint64_t> v1, vector<uint64_t> v2)
{
	chrono::high_resolution_clock::time_point timeStart, timeEnd;
	long p = 44000069; // p=257 > mod 257 > 8 bit int (0-256) must be a prime number
	long r = 1;
	long L = 5; // 6
	long c = 2;
	long w = 64;
	long d = 1;
	long k = 80;
	long s = 0;

	timeStart = chrono::high_resolution_clock::now();
	long m = FindM(k, L, c, p, d, s, 0);
	FHEcontext context(m, p, r);
	buildModChain(context, L, c);

	FHESecKey secretKey(context);
	const FHEPubKey& publicKey = secretKey;
	secretKey.GenSecKey(w); // A Hamming-weight-w secret key
	addSome1DMatrices(secretKey); // compute key-switching matrices that we need, time consuming
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeInitialisation = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Initialisation: " << timeInitialisation.count() << " us" << endl;

	timeStart = chrono::high_resolution_clock::now();
	RR euclidean = helibEuclideanDistance(v1, v2, secretKey, publicKey);
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeDiffEuclidean = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Euclidean: " << euclidean << ", " << timeDiffEuclidean.count() << " us" << endl;

	timeStart = chrono::high_resolution_clock::now();
	RR cosine = helibCosineDistance(v1, v2, secretKey, publicKey);
	timeEnd = chrono::high_resolution_clock::now();
	chrono::microseconds timeDiffCosine = chrono::duration_cast<chrono::microseconds>(timeEnd - timeStart);
	cout << "Cosine: " << cosine << ", " << timeDiffCosine.count() << " us" << endl;

	return make_tuple(timeInitialisation, timeDiffEuclidean, timeDiffCosine);
}
#endif

void writeResult(ofstream &csvFile, int vectorLength, string library, int round, auto result)
{
	csvFile << vectorLength;
	csvFile << ",";
	csvFile << library;
	csvFile << ",";
	//csvFile << round;
	//csvFile << ",";
	csvFile << get<0>(result).count();
	csvFile << ",";
	csvFile << get<1>(result).count();
	csvFile << ",";
	csvFile << get<2>(result).count();
	csvFile << "\n";
	flush(csvFile);
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		cout << "Specify testbed name or special test name (e.g., iot, lbs)" << endl;
		exit(0);
	}
	
	// Log file
	string csvFilename = "performance-smc-testbed-";
	csvFilename += argv[1];
	csvFilename += ".csv";
	cout << csvFilename << endl;
	ofstream csvFile;
	csvFile.open(csvFilename);
	csvFile << "vectorLength,library,initialisation,euclidean,cosine\n";
	
	// Default values
	int evaluationRounds = 10;
	int vectorStepSize = 1000;
	int minVectorLength = 1000;
	int maxVectorLength = 20000;
	// Special cases
	if (strcmp(argv[1], "iot") == 0) {
		vectorStepSize = 50;
		minVectorLength = 200;
		maxVectorLength = 1000;
	} else if (strstr(argv[1], "lbs")) {
		vectorStepSize = 1;
		minVectorLength = 2;
		maxVectorLength = 2;
	}
	
	int testVectorSize = ((maxVectorLength - minVectorLength)  / vectorStepSize) + 1;
	int testVectorLength[testVectorSize];
	for (int i = 0; i < testVectorSize; i++) {
	    testVectorLength[i] = minVectorLength + vectorStepSize * i;
	}
	
	int rand_min = 1, rand_max = 500;
	for(int i = 0; i < testVectorSize; i++) {
		int vectorLength = testVectorLength[i];
		cout << "vector length: " << vectorLength << endl;
		vector<uint64_t> v1;
		vector<uint64_t> v2;
		srand(time(NULL));
		for (int i = 0; i < vectorLength; i++) {
			int val1 = rand()%(rand_max - rand_min + 1) + rand_min;
			int val2 = rand()%(rand_max - rand_min + 1) + rand_min;
			v1.emplace_back(val1);
			v2.emplace_back(val2);
		}
		for (int evaluationRound = 0; evaluationRound < evaluationRounds; evaluationRound++) {
			cout << "---" << endl;
			cout << "round: " << evaluationRound+1 << endl;
			
			#ifdef ENABLE_SEAL
			cout << "seal" << endl;
			auto durationSeal = testSeal(v1, v2);
			writeResult(csvFile, vectorLength, "seal", evaluationRound+1, durationSeal);
			cout << "---" << endl;
			#endif
			
			#ifdef ENABLE_ABY
			cout << "aby" << endl;
			testAby(v1, v2);
			cout << "---" << endl;
			// writeResult(csvFile, vectorLength, "aby", evaluationRound+1, durationAby);
			#endif
			
			#ifdef ENABLE_HELIB
			cout << "helib" << endl;
			auto durationHelib = testHelib(v1, v2);
			writeResult(csvFile, vectorLength, "helib", evaluationRound+1, durationHelib);
			#endif
		}
		cout << "###" << endl;
	}
	csvFile.close();
	return 0;
}
