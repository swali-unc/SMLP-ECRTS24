#pragma once
#include <vector>
#include <random>

#include "task.hpp"
#include "util.hpp"

struct job {
	double start; // The time at which the job started
	double remCost; // The remaining cost of the job
	double deadline; // The absolute deadline of the job
	double totalcost; // The total cost of the job
	int cpu; // The CPU that the job is assigned to (-1 if none)
	request* r; // The request that the job is associated with
};

class request {
public:
	double* Li; // An array of critical section durations if assigned a number of SMs
	int SMcount; // Number of SMs assigned to the request
	job* j; // The job that the request is associated with
	double issuetime; // The time at which the request was issued
	double piblockingtime; // The time for which the request was blocked by a priority inversion

	~request();
};

struct simOutput {
	unsigned int H;
	unsigned int M;
	double Theta;
	unsigned int deadline_miss_count;
	double worst_piblock;
	bool isSMLP;
	double util;
};

typedef unsigned int (*zfunc)(request*, unsigned int, unsigned int);
unsigned int z_omlp(request* r, unsigned int It, unsigned int H);
unsigned int z_smlp(request* r, unsigned int It, unsigned int H);

class gedf_sim {
public:
	gedf_sim(
		char* inFileName,
		unsigned int Hmin,
		unsigned int Hmax,
		unsigned int Hstep,
		double Thetamin,
		double Thetamax,
		double Thetastep,
		double p,
		unsigned int M);
	gedf_sim(
		std::vector<task*>* inTaskSet,
		unsigned int Hmin,
		unsigned int Hmax,
		unsigned int Hstep,
		double Thetamin,
		double Thetamax,
		double Thetastep,
		double p,
		unsigned int M);
	~gedf_sim();

	void runSimulation(int tsIndex, double util, double periodMin, double periodMax);
	void reportResult(simOutput* out);
private:
	unsigned int Hmin;
	unsigned int Hmax;
	unsigned int Hstep;
	double Thetamin;
	double Thetamax;
	double Thetastep;
	double p;
	unsigned int M;
	double u;
	std::default_random_engine gen;

	bool load_task_data(char* inFileName);
	// Returns Lmax and LHmax
	std::tuple<double, double> generate_requests(unsigned int H);
	request* generate_request(task* t, unsigned int H);

	std::vector<task*> tasks;
	std::vector<simOutput*> results;

	SpinLock reportLock;
};

void sim_thread(gedf_sim* sim, unsigned int H, unsigned int M, double Theta, std::vector<task*>* taskSet, bool isSMLP, double hyperperiod);