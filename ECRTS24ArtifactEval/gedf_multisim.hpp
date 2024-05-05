#pragma once

#include <vector>

void runFullSimulation(
	std::vector<double>* tminSet, std::vector<double>* tmaxSet,
	double utilStart, double utilEnd, double utilStep,
	unsigned int Mmin, unsigned int Mmax, unsigned int Mstep,
	unsigned int Hmin, unsigned int Hmax, unsigned int Hstep,
	double thetaMin, double thetaMax, double Tstep, double p,
	int nMax, unsigned int taskSets
);