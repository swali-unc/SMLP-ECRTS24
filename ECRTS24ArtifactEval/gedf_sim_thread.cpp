#include "gedf_sim.hpp"
#include "util.hpp"

#include <vector>
#include <queue>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

#ifndef _MSC_VER
#include <cstring>
#endif

using std::priority_queue;
using std::vector;
using std::deque;

enum acttype {
	RELEASE, UNLOCK, COMPLETE, SLICE
};

struct act {
	acttype type;
	double time;
	job* j;
	request* r;
};

// This let's us iterate through a priority queue
template <class T, class S, class C>
S& PQContainer(priority_queue<T, S, C>& q) {
	struct HackedQueue : private priority_queue<T, S, C> {
		static S& Container(priority_queue<T, S, C>& q) {
			return q.* & HackedQueue::c;
		}
	};
	return HackedQueue::Container(q);
}

bool cmpAct(const act* a, const act* b);
bool cmpEDF(const job* a, const job* b);
bool cmpReq(const request* a, const request* b);

class simData {
public:
	simData(unsigned int H, unsigned int M, double Theta, std::vector<task*>* taskSet, bool isSMLP, double hyperperiod)
		: events(cmpAct), readyQ(cmpEDF), reqPQ(cmpReq)
	{
		this->H = H;
		this->M = M;
		this->Theta = Theta;
		this->taskSet = taskSet;
		this->isSMLP = isSMLP;
		this->hyperperiod = hyperperiod;

		doSim();
	}

	simOutput getOutput() { return out; }
private:
	unsigned int H;
	unsigned int M;
	double Theta;
	std::vector<task*>* taskSet;
	bool isSMLP;
	double hyperperiod;

	// The CPU array, if a CPU is null, then it is idle
	job** cpu;

	// Number of SMs available at time t.
	unsigned int It;
	
	// All of the jobs in the system are in this vector.
	vector<job*> jobs;

	// A priority queue of events, earlier events execute first.
	priority_queue<act*, vector<act*>, decltype(&cmpAct)> events;

	// A priority queue of jobs, a released job with the earliest deadline is prioritized
	priority_queue<job*, vector<job*>, decltype(&cmpEDF)> readyQ;

	// The output results of this simulation are initialized
	simOutput out;

	// The FIFO queue of the SMLP/OMLP
	deque<request*> reqFQ;
	priority_queue<request*, vector<request*>, decltype(&cmpReq)> reqPQ; // PQ of the SMLP/OMLP
	// We don't need an SQ for this simulation, instead, when a request
	//  is satisfied, it pushes a completion event on the event queue
	//  and that is how the job re-enters the ready queue.

	void jobReady(unsigned int& lowestCPU, act* evnt);
	unsigned int getLowestPrioCPU();

	void doSim();

	void satisfy(zfunc z, double t);
};

void simData::doSim() {
	out.H = H;
	out.M = M;
	out.Theta = Theta;
	out.deadline_miss_count = 0;
	out.worst_piblock = 0;
	out.isSMLP = isSMLP;

	// The cpu is a pointer to a job running on that cpu
	//  If the cpu is null, then the cpu is idle
	cpu = new job * [M];
	for (unsigned int i = 0; i < M; ++i)
		cpu[i] = nullptr;

	// Determine our z function.
	// The OMLP allocates the whole GPU (H SMs)
	// The SMLP resizes requests to fit available SMs
	zfunc z = isSMLP ? z_smlp : z_omlp;

	// Add all job releases to the event queue
	for (auto& i : *taskSet) {
		for (double t = 0; t < hyperperiod; t += i->period) {
			auto j = new job{ t, i->cost, t + i->deadline, i->cost, -1, nullptr };
			if (i->r) {
				auto r = new request;
				memcpy(r, i->r, sizeof(request));
				r->j = j;
				r->piblockingtime = 0;
				r->issuetime = -1;
				r->Li = new double[H + 1];
				memcpy(r->Li, i->r->Li, sizeof(double) * (H + 1));
				j->r = r;
			}
			else j->r = nullptr;
			jobs.push_back(j);
			auto a = new act{ RELEASE, t, j, j->r };
			events.push(a);
		}
	}

	// If we are a time-sliced component...
	if (Theta > 0) {
		// Add all time slice events to the event queue
		//  No critical-section can run when a SLICE event happens
		for (double t = 0; t < hyperperiod; t += Theta)
			events.push(new act{ SLICE, t, nullptr, nullptr });
	}

	auto& pqVec = PQContainer(reqPQ); // allows us to iterate through the priority queue

	It = H; // The number of SMs available at t=0.

	//double nextTarget = hyperperiod / 10;
	auto lowestCPU = getLowestPrioCPU();
	for (double t = 0; t < hyperperiod; ) {
		//printf("t: %f\n", t);
		if (!events.size())
			break;
		auto evnt = events.top();
		auto tprev = t;
		bool useEvent = true;

		// First, let's see if we can complete an active job
		//  before the next event happens.
		int smallest_rem = -1;
		for (unsigned int i = 0; i < M; ++i) {
			if (cpu[i] && (smallest_rem < 0 || cpu[i]->remCost < cpu[smallest_rem]->remCost))
				smallest_rem = i;
		}

		// If there is an active job, will it complete before the next event?
		if (smallest_rem >= 0 && cpu[smallest_rem]->remCost < evnt->time - tprev) {
			evnt = new act{ COMPLETE, tprev + cpu[smallest_rem]->remCost, cpu[smallest_rem], nullptr };
			useEvent = false;
		}

		// If we are using the event, then remove it from the event queue
		if (useEvent)
			events.pop();
		t = evnt->time; // Update the new time.
		auto tDelta = t - tprev;

		/*if (t > nextTarget) {
			threadsafe_printf("Simulation at time %f/%f\n", t, hyperperiod);
			nextTarget += hyperperiod / 10;
		}*/

		// Update all pi-blocking in this duration
		// A request is pi-blocked if it is not running
		//  and a job with a lower priority is.
		// NOTE: we don't have to worry about priority-inheritance.
		//  This is because of property 7 in the paper. The only time
		//  PI is applied is for a zero-length duration to complete a request.
		// This makes simulation much easier.
		for (auto& i : reqFQ) {
			if (!cpu[lowestCPU] || cpu[lowestCPU]->deadline < i->j->deadline)
				i->piblockingtime += tDelta;
		}
		for (auto& i : pqVec) {
			if (!cpu[lowestCPU] || cpu[lowestCPU]->deadline < i->j->deadline)
				i->piblockingtime += tDelta;
		}

		// Also update the remaining cost of all jobs that are running.
		for (unsigned int i = 0; i < M; ++i)
			if (cpu[i]) cpu[i]->remCost -= tDelta;

		// Handle the event
		switch (evnt->type) {
		case SLICE:
			// If we are a time-sliced component, then we need to
			//  check if we can satisfy any requests.
			satisfy(z, t);
			break;
		case RELEASE:
			// A job has been released, add it to the job priority queue
			// If the job has a request, it issues the request immediately.
			if (evnt->j->r) {
				evnt->j->r->issuetime = t;
				reqPQ.push(evnt->j->r);
				satisfy(z, t);
			}
			else
				jobReady(lowestCPU, evnt);
			break;
		case UNLOCK:
			// A request has been satisfied, add the job back to the ready queue (or link on cpu)
			//  This also means we might be able to satisfy other requests.

			// New contender for the worst observed pi-blocking?
			if (evnt->r->piblockingtime > out.worst_piblock)
				out.worst_piblock = evnt->r->piblockingtime;
			It += evnt->r->SMcount; // Reclaim SMs that were being used.
			satisfy(z, t); // Try to satisfy more requests

			// Attempt to re-enqueue this job
			jobReady(lowestCPU, evnt);
			break;
		case COMPLETE:
			// A job is complete. Remove it from the CPU.
			if (t > evnt->j->deadline)
				out.deadline_miss_count++;
			if (readyQ.empty())
				cpu[evnt->j->cpu] = nullptr;
			else {
				cpu[evnt->j->cpu] = readyQ.top();
				cpu[evnt->j->cpu]->cpu = evnt->j->cpu;
				readyQ.pop();
			}
			evnt->j->cpu = -1;
			break;
		default:
			threadsafe_printf("Error: Unknown event at time %f\n", t);
		}

		delete evnt;
	}

	delete[] cpu;

	while (events.size()) {
		auto evnt = events.top();
		events.pop();
		delete evnt;
	}

	while (jobs.size()) {
		auto j = jobs.back();
		jobs.pop_back();
		if (j->r)
			delete j->r;
		delete j;
	}

	// Requests are deleted when jobs are deleted
	//while (!reqFQ.empty())
	//	reqFQ.pop_front();
	reqFQ.clear();

	//while (!reqPQ.empty())
	//	reqPQ.pop();
}

void sim_thread(gedf_sim* parent, unsigned int H, unsigned int M, double Theta, std::vector<task*>* taskSet, bool isSMLP, double hyperperiod) {
	auto start = std::chrono::high_resolution_clock::now();
	simData sim(H, M, Theta, taskSet, isSMLP, hyperperiod);
	auto out = sim.getOutput();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	//threadsafe_printf("%s (%lldms): H: %d, M: %d, Theta: %f, Deadline misses: %d, Worst PI-block: %f\n", isSMLP ? "SMLP" : "OMLP", duration, out.H, out.M, out.Theta, out.deadline_miss_count, out.worst_piblock);
	parent->reportResult(&out);
}

void simData::satisfy(zfunc z, double t) {
	// Idea: does the FQ contain less than M requests?
	//  If so, move requests from the PQ to the FQ
	while (!reqPQ.empty() && reqFQ.size() < M) {
		reqFQ.push_back(reqPQ.top());
		reqPQ.pop();
	}

	if (It == 0) return;

	// If the FQ is empty, no requests need to be satisfied
	if (reqFQ.empty())
		return;

	// Find next slice event.
	double nextSlice;
	if (Theta > 0) {
		// Find next slice event.
		nextSlice = ceil(t / Theta) * Theta;
		if( abs(t - nextSlice) < 0.0001 )
			nextSlice += Theta;
	}
	else nextSlice = -1;

	// Go through all requests in the FQ
	for( unsigned int i = 0; i < reqFQ.size(); ) {
		// FZ-blocking when: t + Li >= nextSlice
		auto SMs = z(reqFQ[i], It, H);
		auto completeTime = reqFQ[i]->Li[SMs] + t;
		if (completeTime < nextSlice) {
			// satisfy this request
			reqFQ[i]->SMcount = SMs;
			It -= SMs;
			// remaining cost goes down by the maximum cs len
			//  This is because the WCET is based on the max cs len.
			reqFQ[i]->j->remCost -= reqFQ[i]->Li[1];
			events.push(new act{ UNLOCK, completeTime, reqFQ[i]->j, reqFQ[i] });

			// Is there a request in the PQ that we can add in?
			if (!reqPQ.empty()) {
				reqFQ.push_back(reqPQ.top());
				reqPQ.pop();
			}

			reqFQ.erase(reqFQ.begin() + i);

			// If we are out of SMs, then we are done.
			if (It == 0)
				break;
			continue;
		}
		++i;
	}
}

unsigned int z_omlp(request* r, unsigned int It, unsigned int H) {
	return H;
}

unsigned int z_smlp(request* r, unsigned int It, unsigned int H) {
	unsigned int smallestAllocation = It;

	for (unsigned int i = It; i >= 1; --i) {
		if (r->Li[i] <= r->Li[smallestAllocation])
			smallestAllocation = i;
	}

	return smallestAllocation;
}

bool cmpAct(const act* a, const act* b) {
	return a->time > b->time;
}

bool cmpEDF(const job* a, const job* b) {
	return a->deadline > b->deadline;
}

bool cmpReq(const request* a, const request* b) {
	return a->j->deadline > b->j->deadline;
}

unsigned int simData::getLowestPrioCPU() {
	int lowest = 0;
	for (unsigned int i = 0; i < M; ++i) {
		if (cpu[i] == nullptr)
			return i;
		if (cpu[i]->deadline > cpu[lowest]->deadline)
			lowest = i;
	}

	return lowest;
}

void simData::jobReady(unsigned int& lowestCPU, act* evnt) {
	if (!cpu[lowestCPU] || cpu[lowestCPU]->deadline > evnt->j->deadline) {
		// Run on this CPU, and effectively preempt the other job running.
		if( cpu[lowestCPU] )
			cpu[lowestCPU]->cpu = -1;
		cpu[lowestCPU] = evnt->j;
		evnt->j->cpu = lowestCPU;
		lowestCPU = getLowestPrioCPU(); // recompute
	}
	else
		readyQ.push(evnt->j);
}