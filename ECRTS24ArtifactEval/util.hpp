#pragma once
#include <atomic>

class SpinLock {
	std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
	void lock() {
		while (locked.test_and_set(std::memory_order_acquire)) { ; }
	}
	void unlock() {
		locked.clear(std::memory_order_release);
	}
};

void threadsafe_printf(const char* argv, ...);

void threadsafe_outputResult(
	int tsIndex, unsigned int n, double u,
	double Tmin, double Tmax,
	unsigned int M, unsigned int H, double Theta, bool isSMLP,
	double wpiblock, unsigned int dlmiss);

void threadsafe_openOutputFile(const char* outFileName);
void threadsafe_closeOutputFile();
void threadsafe_flushOutput();