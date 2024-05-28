/* main.cpp
 * 
 * Main file for the artifact evaluation of the ECRTS 2021 submission.
 * This file contains command-line parameter evaluation processing.
 * 
 * The simulation is done in gedf_sim and the task generation is done in task_gen.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "task_gen.hpp"
#include "gedf_sim.hpp"
#include "gedf_multisim.hpp"
#include "util.hpp"

void print_usage(char* binName);

int main(int argc, char* argv[]) {
	if (argc <= 1) {
		print_usage(argv[0]);
		return 0;
	}

	// Generate a task set
	char* outFileName = nullptr;
	unsigned int M = 0; // number of cores
	double Tmin = -1; // periodmin
	double Tmax = -1; // periodmax
	unsigned int n = 0; // number of tasks in a task set
	double U = -1; // utilization

	// Simulate a task set
	char* inFileName = nullptr;
	unsigned int Hmin = 0; // Simulation carried out for H in [Hmin,Hmax] with step size Hstep
	unsigned int Hmax = 0;
	unsigned int Hstep = 0;
	double thetaMin = -1; // Simulation carried out for Theta in [thetaMin,thetaMax] with step size Tstep
	double thetaMax = -1;
	double Tstep = -1;
	double p = -1; // Probability that a task is selected to issue a lock request

	// Do both
	std::vector<double> tminset;
	std::vector<double> tmaxset;
	double utilStart = -1; // Simulation carried out for U in [utilStart,utilEnd] with step size utilStep
	double utilEnd = -1;
	double utilStep = -1;
	unsigned int Mmin = 0; // Simulation carried out for M in [Mmin,Mmax] with step size Mstep
	unsigned int Mmax = 0;
	unsigned int Mstep = 0;
	unsigned int taskSets = 0; // Number of task sets to generate

	// Populate the input parameters for task set generation
	for (int i = 2; i + 1 < argc; i += 2) {
		if (!strcmp(argv[i], "-o"))
			outFileName = argv[i + 1];
		else if (!strcmp(argv[i], "-m"))
			M = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-tmin"))
			Tmin = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-tmax"))
			Tmax = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-n"))
			n = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-u"))
			U = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-i"))
			inFileName = argv[i + 1];
		else if (!strcmp(argv[i], "-Hmin"))
			Hmin = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-Hmax"))
			Hmax = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-Hstep"))
			Hstep = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-Tmin"))
			thetaMin = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-Tmax"))
			thetaMax = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-Tstep"))
			Tstep = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-p"))
			p = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-tminset")) {
			while (i + 1 < argc && argv[i + 1][0] != '-') {
				tminset.push_back(atof(argv[i + 1]));
				++i;
			}
			--i; // to fix the loop
		}
		else if (!strcmp(argv[i], "-tmaxset")) {
			while (i + 1 < argc && argv[i + 1][0] != '-') {
				tmaxset.push_back(atof(argv[i + 1]));
				++i;
			}
			--i; // to fix the loop
		}
		else if (!strcmp(argv[i], "-utilMin"))
			utilStart = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-utilMax"))
			utilEnd = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-utilStep"))
			utilStep = atof(argv[i + 1]);
		else if (!strcmp(argv[i], "-Mmin"))
			Mmin = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-Mmax"))
			Mmax = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-Mstep"))
			Mstep = atoi(argv[i + 1]);
		else if (!strcmp(argv[i], "-taskSets"))
			taskSets = atoi(argv[i + 1]);
		else {
			printf("Unknown parameter: %s\n", argv[i]);
			return 0;
		}
	}

	// stores the error messages
	char errMsg[512];
	errMsg[0] = '\0';

	if (outFileName == nullptr)
		strcat(errMsg, "Missing output filename\n");

	if (!strcmp(argv[1], "-t")) {
		// Were any parameters missing?
		if (Tmin <= 0)
			strcat(errMsg, "Missing minimum period\n");
		if (Tmax <= 0)
			strcat(errMsg, "Missing maximum period\n");
		if( n == 0 )
			strcat(errMsg, "Missing number of tasks\n");
		if( U <= 0 )
			strcat(errMsg, "Missing target utilization\n");
		if (Tmin >= Tmax)
			strcat(errMsg, "Tmin must be less than Tmax\n");
		if (M == 0)
			strcat(errMsg, "Missing number of cores\n");

		if (errMsg[0] != '\0') {
			printf("%s", errMsg);
			return 0;
		}

		// Generate the task set
		printf("Generating task set..\n");
		TaskGen tg(M, Tmin, Tmax, n, U);
		tg.outputTaskSet(outFileName);
		printf("Output to %s\n", outFileName);
	}
	else if (!strcmp(argv[1], "-s")) {
		// Were any parameters missing?
		if (inFileName == nullptr)
			strcat(errMsg, "Missing input filename\n");
		if (Hmin == 0)
			strcat(errMsg, "Missing minimum number of SMs\n");
		if (Hmax == 0)
			strcat(errMsg, "Missing maximum number of SMs\n");
		if (Hstep == 0)
			strcat(errMsg, "Missing step size for number of SMs\n");
		if (thetaMin <= 0)
			strcat(errMsg, "Missing minimum Theta-time slice duration\n");
		if (thetaMax <= 0)
			strcat(errMsg, "Missing maximum Theta-time slice duration\n");
		if (Tstep <= 0)
			strcat(errMsg, "Missing step size for Theta-time slice duration\n");
		if (p <= 0)
			strcat(errMsg, "Missing probability to select a task to issue a lock req\n");
		if (Hmin > Hmax)
			strcat(errMsg, "Hmin must be leq than Hmax\n");
		if (thetaMin > thetaMax)
			strcat(errMsg, "thetaMin must be leq thetaMax\n");
		if (M == 0)
			strcat(errMsg, "Missing number of cores\n");

		if (errMsg[0] != '\0') {
			printf("%s", errMsg);
			return 0;
		}

		// Simulate the task set
		threadsafe_openOutputFile(outFileName);
		gedf_sim sim(inFileName, Hmin, Hmax, Hstep, thetaMin, thetaMax, Tstep, p, M);
		sim.runSimulation(1,-1,-1,-1);
		threadsafe_closeOutputFile();
	}
	else if (!strcmp(argv[1], "-a")) {
		if (n == 0)
			strcat(errMsg, "Missing number of tasks\n");
		if (Hmin == 0)
			strcat(errMsg, "Missing minimum number of SMs\n");
		if (Hmax == 0)
			strcat(errMsg, "Missing maximum number of SMs\n");
		if (Hstep == 0)
			strcat(errMsg, "Missing step size for number of SMs\n");
		if (thetaMin <= 0)
			strcat(errMsg, "Missing minimum Theta-time slice duration\n");
		if (thetaMax <= 0)
			strcat(errMsg, "Missing maximum Theta-time slice duration\n");
		if (Tstep <= 0)
			strcat(errMsg, "Missing step size for Theta-time slice duration\n");
		if (p <= 0)
			strcat(errMsg, "Missing probability to select a task to issue a lock req\n");
		if (tminset.size() == 0)
			strcat(errMsg, "Missing minimum period set\n");
		if (tmaxset.size() == 0)
			strcat(errMsg, "Missing maximum period set\n");
		if (tminset.size() != tmaxset.size())
			strcat(errMsg, "Mismatched period set sizes\n");
		if (utilStart < 0)
			strcat(errMsg, "Missing start utilization\n");
		if (utilEnd < 0)
			strcat(errMsg, "Missing end utilization\n");
		if (utilStep < 0)
			strcat(errMsg, "Missing utilization step\n");
		if (Mmin == 0)
			strcat(errMsg, "Missing minimum number of cores\n");
		if (Mmax == 0)
			strcat(errMsg, "Missing maximum number of cores\n");
		if (Mstep == 0)
			strcat(errMsg, "Missing step size for number of cores\n");
		if (Mmin > Mmax)
			strcat(errMsg, "Mmin must be leq than Mmax\n");
		if (taskSets == 0)
			strcat(errMsg, "Missing number of task sets\n");

		if (errMsg[0] != '\0') {
			printf("%s", errMsg);
			return 0;
		}

		// Run multiple simulations
		threadsafe_openOutputFile(outFileName);
		runFullSimulation(&tminset, &tmaxset, utilStart, utilEnd, utilStep, Mmin, Mmax, Mstep, Hmin, Hmax, Hstep, thetaMin, thetaMax, Tstep, p, n, taskSets);
		threadsafe_closeOutputFile();
	}
	else {
		print_usage(argv[0]);
	}

	return 0;
}

void print_usage(char* binName) {
	printf("%s\nUsage: (first argument must be either -t or -s or -a)\n", binName);
	printf(
		"\t-t\tGenerate Task Set\n"
			"\t\t-o [filename]\tOutput csv filename\n"
			"\t\t-m [coreCount]\tNumber of cores in component\n"
			"\t\t-tmin [minPeriod]\tMinimum period (ms)\n"
			"\t\t-tmax [maxPeriod]\tMaximum period (ms)\n"
			"\t\t-n [numTasks]\tNumber of tasks\n"
			"\t\t-u [targetUtil]\tTarget Normalized Utilization (0,1)\n"
		"\t-s\tRun simulation\n"
			"\t\t-i [filename]\tInput task set csv\n"
			"\t\t-o [filename]\tOutput csv filename of results\n"
			"\t\t-Hmin [Hmin]\tNumber of min SMs given to component\n"
			"\t\t-Hmax [Hmax]\tNumber of max SMs given to component\n"
			"\t\t-Hstep [Hstep]\tHow to modify H every simulation\n"
			"\t\t-Tmin [ThetaMin]\tMin Theta-time slice duration\n"
			"\t\t-Tmax [ThetaMax]\tMax Theta-time slice duration\n"
			"\t\t-Tstep [ThetaStep]\tAmount to change Theta every simulation\n"
			"\t\t-p [prob]\tProbability to select a task to issue a lock req\n"
			"\t\t-m [coreCount]\tNumber of cores in component\n"
		"\t-a\tRun multiple simulations\n"
			"\t\t-o [filename]\tOutput csv filename\n"
			"\t\t-Mmin [coreCount]\tStarting number of cores in component\n"
			"\t\t-Mmax [coreCount]\tStarting number of cores in component\n"
			"\t\t-Mstep [coreCount]\tStarting number of cores in component\n"
			"\t\t-tminset [tmin1, tmin2, ...]\tMinimum periods (ms)\n"
			"\t\t-tmaxset [tmax1, tmax2, ...]\tMaximum period (ms)\n"
			"\t\t-n [numTasks]\tGenerate from U~(2m,numTasks) tasks\n"
			"\t\t-taskSets [numTaskSets]\tThe number of task sets to generate\n"
			"\t\t-utilMin [minUtil]\tTarget Start Normalized Utilization (0,1)\n"
			"\t\t-utilMax [maxUtil]\tTarget End Normalized Utilization (0,1)\n"
			"\t\t-utilStep [utilStep]\tUtilization step\n"
			"\t\t-Hmin [Hmin]\tNumber of min SMs given to component\n"
			"\t\t-Hmax [Hmax]\tNumber of max SMs given to component\n"
			"\t\t-Hstep [Hstep]\tHow to modify H every simulation\n"
			"\t\t-Tmin [ThetaMin]\tMin Theta-time slice duration\n"
			"\t\t-Tmax [ThetaMax]\tMax Theta-time slice duration\n"
			"\t\t-Tstep [ThetaStep]\tAmount to change Theta every simulation\n"
			"\t\t-p [prob]\tProbability to select a task to issue a lock req\n"
		"\n"
	);
}