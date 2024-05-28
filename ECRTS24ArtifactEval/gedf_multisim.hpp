#pragma once

/* This function will generate multiple results at the same time.
 * It is a "full" simulation in that it will generate simulation scenarios
 *  given the parameters, and then run the scenarios on multiple threads.
 */

#include <vector>

// tminSet and tmaxSet are vectors of Tmin and Tmax values to use.
// utilStart, utilEnd, utilStep are the range of utilization values to use.
// Mmin, Mmax, Mstep are the range of M values to use.
// Hmin, Hmax, Hstep are the range of H values to use.
// thetaMin, thetaMax, Tstep are the range of Theta values to use.
// p is the probability of a task issuing a request
// nMax is the maximum number of tasks to generate in a task set
// taskSets is the number of task sets to generate for each scenario
void runFullSimulation(
	std::vector<double>* tminSet, std::vector<double>* tmaxSet,
	double utilStart, double utilEnd, double utilStep,
	unsigned int Mmin, unsigned int Mmax, unsigned int Mstep,
	unsigned int Hmin, unsigned int Hmax, unsigned int Hstep,
	double thetaMin, double thetaMax, double Tstep, double p,
	int nMax, unsigned int taskSets
);