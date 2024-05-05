#pragma once
#include <vector>
#include <random>

#include "task.hpp"

class TaskGen {
public:
	TaskGen( unsigned int M, // number of processors guaranteed to component
		double Tmin, // minimum period
		double Tmax, // maximum period
		unsigned int n, // number of tasks
		double U); // target utilization
	~TaskGen();

	void outputTaskSet(std::vector<task*>* tasksOut) const;
	void outputTaskSet(char* outCsvFileName) const;
	double getActualUtilization() const { return actualU; }
private:
	unsigned int M; // number of processors guaranteed to component
	double Tmin; // minimum period
	double Tmax; // maximum period
	unsigned int n; // number of tasks
	double U; // target utilization
	std::default_random_engine gen;
	double actualU; // actual utilization

	// List of tasks. Cleaned up in destructor.
	std::vector<task*> tasks;

	void generateTaskSet();
	double* generate_utilizations();

	double sample_Ti();
	double sample_ri();
};