#include "gedf_multisim.hpp"
#include "task_gen.hpp"
#include "gedf_sim.hpp"
#include "util.hpp"

#include <random>
#include <vector>
#include <thread>

/* How simulations are run:
 *  First, we pick a value for U in [utilStart, utilEnd] with step utilStep.
 *  Second, we pick a value for M in [Mmin, Mmax] with step Mstep.
 *  Third, we pick a value for n~U(2M, nMax).
 *	Fourth, we pick a value for Tmin and Tmax from the given sets.
 *  Fifth, we generate a "taskSets" number of task sets with the given parameters.
 *  Sixth, we create a task set and a sim_entry thread, which calls gedf_sim::runSimulation.
 *   This data is copied to the simulation, and thus the task set can be deleted immediately after.
 *   We could likely optimize this by passing the task set data and cleaning up appropriately.
 * 
 * The following has not yet been selected:
 *  Which tasks issue requests, Theta, and what is the value of H.
 *  These values are selected in the gedf_sim code.
 * 
 * Finally, gedf_sim::runSimulation is called on up to a PARALLEL_SETS number of threads.
 *  Once these threads finish, the next threads start, assuming there are more task sets to generate.
 *  Note that gedf_sim also threads multiple threads, for a total of 2*|ThetaValues|*PARALLEL_SETS number of threads.
 */

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

	// We start at high utilization values and work our way down.
	// There is no reason for this.
	for (double U = utilEnd; U >= utilStart; U -= utilStep) {
		for (unsigned int M = Mmin; M <= Mmax; M += Mstep) {
			unsigned int n; // The number of tasks in the task set
		
			// ensure dist object is cleaned up by scoping
			{
				std::uniform_int_distribution<int> dist(2 * M, nMax);
				n = std::max((unsigned int)dist(gen), 2*M);
			}

			// For each Tmin and Tmax...
			for (unsigned int Tidx = 0; Tidx < tminSet->size(); ++Tidx) {
				double Tmin = tminSet->at(Tidx);
				double Tmax = tmaxSet->at(Tidx);

				// Generate our task sets
				for (unsigned int taskIdx = 0; taskIdx < taskSets; taskIdx += PARALLEL_SETS) {
					std::vector<gedf_sim*> sims;
					std::vector<std::thread> threads;

					threadsafe_printf("Start: %d/%d %f %d %f %f\n", taskIdx, taskSets, U, M, Tmin, Tmax);

					// Create up to PARALLEL_SETS number of threads to handle these task sets.
					for (unsigned int i = 0; i < PARALLEL_SETS; ++i) {
						// Generate our tasks
						TaskGen tg(M, Tmin, Tmax, n, U);
						std::vector<task*>* taskSet = new std::vector<task*>();
						tg.outputTaskSet(taskSet); // Grab the raw task data

						// Create the simulation
						auto gs = new gedf_sim(
							taskSet, Hmin, Hmax, Hstep,
							thetaMin, thetaMax, thetaStep, p, M
						);
						sims.push_back(gs);

						// Create the thread
						threads.push_back(
							std::thread(sim_entry, gs, taskIdx + i, U, Tmin, Tmax)
						);

						// Delete task sets (they are copied to the simulation)
						for (auto& i : *taskSet)
							delete i;
						delete taskSet;
					}

					// Wait for all PARALLEL_SETS threads to finish.
					// This could be optimized to start the next thread as soon as one finishes,
					//  but having some small amount of laxity is fine to not burn out the CPU over however many hours/days.
					for (auto& i : threads)
						i.join();
					for (auto& i : sims)
						delete i;
					threads.clear();
					sims.clear();

					// If there is a reason to stop the sim, at least we will have some results by
					// flushing the output every so often.
					threadsafe_flushOutput();
				}
			}
		}
	}

	threadsafe_printf("Done with simulations.\n");
}