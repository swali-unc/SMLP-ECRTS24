#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#else
#include <cstring>
#endif

#include "gedf_sim.hpp"

#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <numeric>
#include <thread>
#include <queue>

using std::fstream;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::thread;
using std::tuple;
using std::tie;

void gedf_sim::runSimulation(int tsIndex, double util, double Tmin, double Tmax) {
	int hyperperiod = std::accumulate(tasks.begin(), tasks.end(), 1, [](int a, task* b) { return std::lcm(a, (int)b->period); });

	if (util < 0)
		util = u;

	for (unsigned int H = Hmin; H <= Hmax; H += Hstep) {
		// We need to determine which tasks issue requests.
		//  Request is defined with a known H, so each H value needs a new set of requests.
		double LHmax, Lmax;
		tie(Lmax, LHmax) = generate_requests(H);

		std::vector<std::thread> threads;

		for (double Theta = Thetamin; Theta <= Thetamax; Theta += Thetastep) {
			if (LHmax > Theta)
				;// threadsafe_printf("Warning: LHMax > Theta (%f > %f)\n", LHmax, Theta);
			else if (Lmax > Theta)
				;// threadsafe_printf("Warning: Lmax > Theta (%f > %f)\n", Lmax, Theta);
			else {
				threads.push_back(thread(sim_thread, this, H, M, Theta, &tasks, true, hyperperiod));
				threads.push_back(thread(sim_thread, this, H, M, Theta, &tasks, false, hyperperiod));
			}
		}

		// join all the threads
		for (auto& i : threads)
			i.join();

		for (auto i : results) {
			threadsafe_outputResult(tsIndex,(unsigned int)tasks.size(), util, Tmin, Tmax, i->M, i->H, i->Theta, i->isSMLP, i->worst_piblock, i->deadline_miss_count);
			delete i;
		}
		results.clear();
	}

}

void gedf_sim::reportResult(simOutput* out) {
	reportLock.lock();
	results.push_back(new simOutput{out->H, out->M, out->Theta,
		out->deadline_miss_count, out->worst_piblock, out->isSMLP, out->util});
	reportLock.unlock();
}

tuple<double,double> gedf_sim::generate_requests(unsigned int H) {
	uniform_real_distribution<double> dist(0, 1);
	double LHmax = 0;
	double Lmax = 0;
	auto maxf = [](double a, double b) { return std::max(a, b); };
	//std::priority_queue<double> amaxes;
	//const unsigned int h = 1;

	for (auto& i : tasks) {
		if (i->r)
			delete i->r;

		if (dist(gen) < p) {
			i->r = generate_request(i, H);
			Lmax = std::accumulate(i->r->Li, i->r->Li + H + 1, Lmax, maxf);
			LHmax = std::max(LHmax, i->r->Li[H]);
			/*double largestAmax = h * i->r->Li[h];
			for (unsigned int j = h; j < H; ++j) {
				if (j * i->r->Li[j] > largestAmax && (j == h || std::abs(i->r->Li[j] - i->r->Li[j - 1]) > 0.001))
					largestAmax = j * i->r->Li[j];
			}
			amaxes.push(largestAmax);*/
		}
		else
			i->r = nullptr;
	}

	/*std::vector<double> amaxvec;
	for (unsigned int i = 0; i < M; ++i) {
		amaxvec.push_back(amaxes.top());
		amaxes.pop();
	}
	amaxes = std::priority_queue<double>();
	double Bfq = 0;
	for (unsigned int i = 0; i < M - 1; ++i)
		Bfq += amaxvec[i];
	Bfq = (Bfq / H) + Lmax;
	threadsafe_printf("X= %f, 2M-1= %f (%f)\n", Bfq * 2, (2 * M - 1) * Lmax, (2 * M - 1) * LHmax);*/

	return { Lmax, LHmax };
}

bool gedf_sim::load_task_data(char* inFileName) {
	fstream fin;
	fin.open(inFileName, std::ios::in);

	// The file headers of the CSV file aren't useful data
	{
		std::string s;
		getline(fin, s);
	}

	u = 0;

	// Read the data from the CSV
	while (!fin.eof()) {
		task* t;
		std::string cost, period, deadline;
		getline(fin, cost, ',');

		if (cost.empty())
			break;

		getline(fin, period, ',');
		getline(fin, deadline);

		t = new task();
		t->cost = std::stod(cost);
		t->period = std::stod(period);
		t->deadline = std::stod(deadline);
		t->r = nullptr; // We will determine requests later
		tasks.push_back(t);

		u += (t->cost / t->period);
	}

	fin.close();
	return true;
}

request* gedf_sim::generate_request(task* t, unsigned int H) {
	request* r = new request();
	r->Li = new double[H + 1];
	memset(r->Li, 0, sizeof(double) * (H + 1));
	r->SMcount = 0;
	r->j = nullptr;

	uniform_real_distribution<double> dist_cost(0, t->cost);
	uniform_int_distribution<int> dist_sm(1, H);
	int rho = dist_sm(gen); // rho is the parallelism level that optimally satisfies the GPU kernel
	r->Li[1] = dist_cost(gen); // The highest cost is when only running on 1 SM

	// This is the equation given in the experimental results in the paper.
	//  It generates a step graph.
	for (int i = 1; (unsigned int)i <= H; ++i)
		r->Li[i] = r->Li[1] * std::max((double)1, ceil(((double)rho - i + 1) / (double)(i))) / (double)rho;

	r->issuetime = -1;
	r->piblockingtime = 0;
	return r;
}

gedf_sim::gedf_sim(
	char* inFileName,
	unsigned int Hmin,
	unsigned int Hmax,
	unsigned int Hstep,
	double Thetamin,
	double Thetamax,
	double Thetastep,
	double p,
	unsigned int M) {
	this->Hmin = Hmin;
	this->Hmax = Hmax;
	this->Hstep = Hstep;
	this->Thetamin = Thetamin;
	this->Thetamax = Thetamax;
	this->Thetastep = Thetastep;
	this->p = p;
	this->M = M;
	gen.seed((unsigned int)time(nullptr));

	if (!load_task_data(inFileName))
		throw std::invalid_argument("Failed to load task data");
}

gedf_sim::gedf_sim(
	std::vector<task*>* inTaskSet,
	unsigned int Hmin,
	unsigned int Hmax,
	unsigned int Hstep,
	double Thetamin,
	double Thetamax,
	double Thetastep,
	double p,
	unsigned int M) {
	this->Hmin = Hmin;
	this->Hmax = Hmax;
	this->Hstep = Hstep;
	this->Thetamin = Thetamin;
	this->Thetamax = Thetamax;
	this->Thetastep = Thetastep;
	this->p = p;
	this->M = M;
	gen.seed((unsigned int)time(nullptr));

	// need to make a copy of the task set, because other threads will probably
	//  also be using the task set.
	u = 0;
	for (auto& i : *inTaskSet) {
		tasks.push_back(new task{ i->cost, i->period, i->deadline, nullptr });
		u += (i->cost / i->period);
	}
}

gedf_sim::~gedf_sim() {
	for (auto& i : tasks) {
		if (i->r)
			delete i->r;
		delete i;
	}

	for (auto& i : results)
		delete i;
}

request::~request() {
	if (Li)
		delete[] Li;
}