#include <math.h>
#include <cassert>
#include <thread>
#include <ENCRYPTO_utils/crypto/crypto.h>
#include "abycore/sharing/sharing.h"
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/arithmeticcircuits.h"
#include "abycore/circuit/circuit.h"
using namespace std;

// cmake .
// make

share* BuildInnerProductCircuit(share *s_x, share *s_y, uint32_t num, ArithmeticCircuit *ac) {
	s_x = ac->PutMULGate(s_x, s_y);
	s_x = ac->PutSplitterGate(s_x);
	for (uint32_t i = 1; i < num; i++) {
		s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	}
	s_x->set_bitlength(1);
	return s_x;
}

share* BuildEuclideanDistance(share *s_x, share *s_y, uint32_t num, ArithmeticCircuit *ac) {
	s_x = ac->PutSUBGate(s_x, s_y);
	s_x = ac->PutMULGate(s_x, s_x);
	s_x = ac->PutSplitterGate(s_x);
	for (uint32_t i = 1; i < num; i++) {
		s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	}
	s_x->set_bitlength(1);
	return s_x;
}

int32_t test(e_role role, const string& address, uint16_t port, seclvl seclvl,
		uint32_t bitlen, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing) {
	uint32_t num = 3;
	ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg);
	vector<Sharing*>& sharings = party->GetSharings();
	ArithmeticCircuit* circ = (ArithmeticCircuit*) sharings[sharing]->GetCircuitBuildRoutine();
	//Circuit* circ = sharings[sharing]->GetCircuitBuildRoutine();

	uint64_t x, y;
	uint64_t min = 1, max = 500;
	uint64_t euclidean, sum_xx = 0, sum_xy = 0, sum_yy = 0;
	share *s_x_vec, *s_y_vec, *s_out;
	share *s_out_xy, *s_out_xx, *s_out_yy;

	uint64_t * xvals = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t * yvals = (uint64_t*) malloc(num * sizeof(uint64_t));

	srand(time(NULL));
	for (uint16_t i = 0; i < num; i++) {
		x = rand()%(max-min + 1) + min;
		y = rand()%(max-min + 1) + min;
		sum_xy += x*y;
		sum_xx += x*x;
		sum_yy += y*y;
		xvals[i] = x;
		yvals[i] = y;
	}

	s_x_vec = circ->PutSIMDINGate(num, xvals, bitlen, SERVER);
	s_y_vec = circ->PutSIMDINGate(num, yvals, bitlen, CLIENT);

	//BooleanCircuit* boolCirc = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
	//share* sum = boolCirc->PutFPGate(s_x_vec, s_y_vec, ADD, num, no_status);

	//s_out = BuildInnerProductCircuit(s_x_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out = BuildEuclideanDistance(s_x_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out = circ->PutOUTGate(s_out, ALL);

	s_out_xy = BuildInnerProductCircuit(s_x_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out_xy = circ->PutOUTGate(s_out_xy, ALL);
	s_out_xx = BuildInnerProductCircuit(s_x_vec, s_x_vec, num, (ArithmeticCircuit*) circ);
	s_out_xx = circ->PutOUTGate(s_out_xx, ALL);
	s_out_yy = BuildInnerProductCircuit(s_y_vec, s_y_vec, num, (ArithmeticCircuit*) circ);
	s_out_yy = circ->PutOUTGate(s_out_yy, ALL);

	//share* sqrt_val = boolCirc->PutFPGate(s_out_xy, SQRT, num, no_status);
	//sqrt_val = boolCirc->PutOUTGate(sqrt_val, ALL);
	//uint64_t val = sqrt_val->get_clear_value<uint64_t>();
	//cout << "sqrt: " << val << endl;

	party->ExecCircuit();

	euclidean = s_out->get_clear_value<uint64_t>();

	uint64_t out_xy = s_out_xy->get_clear_value<uint64_t>();
	uint64_t out_xx = s_out_xx->get_clear_value<uint64_t>();
	uint64_t out_yy = s_out_yy->get_clear_value<uint64_t>();
	//cout << "xy: " << out_xy << endl;
	//cout << "xx: " << out_xx << endl;
	//cout << "yy: " << out_yy << endl;
	double cosine = 1.0;
	cosine -= out_xy / (sqrt(out_xx) * sqrt(out_yy));

	cout << "Cosine: " << cosine << endl;
	cout << "Euclidean: " << sqrt(euclidean) << endl;

	for (int i=0; i<num; i++) {
		if (i!=0) {
			cout << ", ";
		}
		cout << xvals[i];
	}
	cout << endl;
	for (int i=0; i<num; i++) {
		if (i!=0) {
			cout << ", ";	
		}
		cout << yvals[i];
	}
	cout << endl;

	delete s_x_vec;
	delete s_y_vec;
	delete party;

	return 0;
}

int main(int argc, char** argv)
{
	//e_sharing sharing = S_YAO;
	e_sharing sharing = S_ARITH;
	uint32_t bitlen = 64; // 16, 32, 64
	uint32_t secparam = 128;
	uint32_t nthreads = 1;

	uint16_t port = 7766;
	string address = "127.0.0.1";
	e_mt_gen_alg mt_alg = MT_OT;

	seclvl seclvl = get_sec_lvl(secparam);
	thread server(test, SERVER, address, port, seclvl, bitlen, nthreads, mt_alg, sharing);
	thread client(test, CLIENT, address, port, seclvl, bitlen, nthreads, mt_alg, sharing);
	server.join();
	client.join();
	return 0;
}
