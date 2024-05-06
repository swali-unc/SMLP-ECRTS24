#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#else
#include <ctime>
#endif

#include "task_gen.hpp"

#include <cmath>

using std::uniform_real_distribution;

// Period must be minimally divisible by this value
constexpr double tg = 1.0;

// Emberson et. al. 2010, "Techniques for the synthesis of multiprocessor tasksets" WATERS'10
void TaskGen::generateTaskSet() {
	double* utils = generate_utilizations();
	actualU = 0;

	// This procedure is exactly as described in the Emberson paper
	// Goal will be to get per-task utilization values that add up
	// to the target utilization U. Will not always be possible but
	// will be close. Higher values of n help reduce this error.
	for (unsigned int i = 0; i < n; ++i) {
		auto Ti = sample_Ti();
		// Note the min will prevent the cost from exceeding the period/deadline
		//  This is why the utilization is not exactly U, but is close with
		//  higher values of n.
		auto t = new task{ std::min(M * utils[i] * Ti, Ti), Ti, Ti, nullptr };
		tasks.push_back(t);
		actualU += t->cost / t->period;
	}

	delete[] utils;
}

double* TaskGen::generate_utilizations() {
	double* r = new double[n];

	// Generate a utilization from 0 to 1 for each task
	{
		uniform_real_distribution<double> dist(0, 1);
		for (unsigned int i = 1; i < n; ++i)
			r[i] = dist(gen);
	}

	// Rebalance the utilizations to achieve the correct total utilization
	double* s = new double[n + 1];
	s[0] = 0;
	s[n] = U;

	for (int i = n; i > 1; --i)
		s[i - 1] = s[i] * powl(r[i - 1], (double)1 / (double)(i - 1));
	delete[] r;

	double* ui = new double[n];
	for (unsigned int i = 0; i < n; ++i)
		ui[i] = s[i + 1] - s[i];
	delete[] s;

	return ui;
}

double TaskGen::sample_Ti() {
	return floor(exp(sample_ri()) / tg) * tg;
}

double TaskGen::sample_ri() {
	uniform_real_distribution<double> dist(log10(Tmin), log10(Tmax + tg));
	return dist(gen);
}

void TaskGen::outputTaskSet(char* outCsvFileName) const {
	FILE* f = fopen(outCsvFileName, "w");
	if (!f) {
		printf("Error opening file %s\n", outCsvFileName);
		return;
	}

	fprintf(f, "cost,period,deadline\n");
	for(auto& i : tasks)
		fprintf(f, "%f,%f,%f\n", i->cost, i->period, i->deadline);
	fclose(f);
}

void TaskGen::outputTaskSet(std::vector<task*>* tasksOut) const {
	for (auto& i : tasks) {
		tasksOut->push_back(new task { i->cost, i->period, i->deadline, nullptr });
	}
}

TaskGen::TaskGen(unsigned int M,
	double Tmin,
	double Tmax,
	unsigned int n,
	double U) {
	this->M = M;
	this->Tmin = Tmin;
	this->Tmax = Tmax;
	this->n = n;
	this->U = U;
	gen.seed((unsigned int)time(nullptr));

	generateTaskSet();
}

TaskGen::~TaskGen() {
	for (auto& i : tasks)
		delete i;
}