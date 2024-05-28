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
	unsigned int H; // For what value of H did we get these results for?
	unsigned int M; // For what value of M...
	double Theta; // For what value of Theta...
	unsigned int deadline_miss_count; // How many deadlines were missed?
	double worst_piblock; // What was the worst observed pi-blocking duration?
	bool isSMLP; // Was this for the SMLP or the OMLP?
	double util; // For what value of U...
};

// This is the z function. Given a request, the number of available SMs, and the total SMs in the component.
//  See z_omlp and z_smlp for examples.
typedef unsigned int (*zfunc)(request*, unsigned int, unsigned int);

// Always returns H (lock the entire GPU)
unsigned int z_omlp(request* r, unsigned int It, unsigned int H);

// Returns the smallest number of SMs that can satisfy this request while not reducing the kernel duration below
// locking all available SMs, given by It.
unsigned int z_smlp(request* r, unsigned int It, unsigned int H);

class gedf_sim {
public:
	// If the task set is in a file, we need to specify it here
	// alongside system parameters to run the simulation.
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

	// If the task set is passed as a vector, then we
	// copy that data into this object.
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

	// Creates two simulation threads, for the OMLP and the SMLP, per value of Theta.
	// Note that util, periodMin, periodMax must be -1 if the task set is loaded from a file.
	void runSimulation(int tsIndex, double util, double periodMin, double periodMax);

	// The thread reports back to the parent object with this thread-safe method.
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

	// Returns Lmax and LHmax, useful for comparing analytical OMLP and SMLP
	std::tuple<double, double> generate_requests(unsigned int H);

	// Creates a request for a task.
	request* generate_request(task* t, unsigned int H);

	std::vector<task*> tasks;

	// All of the results from the simulation are stored here.
	std::vector<simOutput*> results;

	// Reporting back to the parent object is thread-safe with this spinlock.
	SpinLock reportLock;
};

void sim_thread(gedf_sim* sim, unsigned int H, unsigned int M, double Theta, std::vector<task*>* taskSet, bool isSMLP, double hyperperiod);