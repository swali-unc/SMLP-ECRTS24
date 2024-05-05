#include "gedf_multisim.hpp"
#include "task_gen.hpp"
#include "gedf_sim.hpp"
#include "util.hpp"

#include <random>
#include <vector>
#include <thread>

#define PARALLEL_SETS 5

void sim_entry(gedf_sim* sim, unsigned int taskIdx, double U, double Tmin, double Tmax) {
	sim->runSimulation(taskIdx, U, Tmin, Tmax);
}

void runFullSimulation(
	std::vector<double>* tminSet, std::vector<double>* tmaxSet,
	double utilStart, double utilEnd, double utilStep,
	unsigned int Mmin, unsigned int Mmax, unsigned int Mstep,
	unsigned int Hmin, unsigned int Hmax, unsigned int Hstep,
	double thetaMin, double thetaMax, double thetaStep, double p,
	int nMax, unsigned int taskSets
) {
	std::default_random_engine gen;
	for (double U = utilEnd; U >= utilStart; U -= utilStep) {
		for (unsigned int M = Mmin; M <= Mmax; M += Mstep) {
			unsigned int n;
		
			{
				std::uniform_int_distribution<int> dist(2 * M, nMax);
				n = std::max(dist(gen), 100);
			}

			for (unsigned int Tidx = 0; Tidx < tminSet->size(); ++Tidx) {
				double Tmin = tminSet->at(Tidx);
				double Tmax = tmaxSet->at(Tidx);

				for (unsigned int taskIdx = 0; taskIdx < taskSets; ) {
					std::vector<gedf_sim*> sims;
					std::vector<std::thread> threads;

					threadsafe_printf("Start: %d/%d %f %d %f %f\n", taskIdx, taskSets, U, M, Tmin, Tmax);
					for (unsigned int i = 0; i < PARALLEL_SETS; ++i) {
						TaskGen tg(M, Tmin, Tmax, n, U);
						std::vector<task*>* taskSet = new std::vector<task*>();
						tg.outputTaskSet(taskSet);

						auto gs = new gedf_sim(
							taskSet, Hmin, Hmax, Hstep,
							thetaMin, thetaMax, thetaStep, p, M
						);
						sims.push_back(gs);
						threads.push_back(
							std::thread(sim_entry, gs, taskIdx + i, U, Tmin, Tmax)
						);

						for (auto& i : *taskSet)
							delete i;
						delete taskSet;
					}

					for (auto& i : threads)
						i.join();
					for (auto& i : sims)
						delete i;
					threads.clear();
					sims.clear();

					taskIdx += PARALLEL_SETS;
				}
			}
		}
	}

	threadsafe_printf("Done with simulations.\n");
}