#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "util.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
int vasprintf(char** strp, const char* fmt, va_list ap) {
	int len = _vscprintf(fmt, ap);
	if (len == -1) return -1;

	*strp = (char*)malloc(len + 1);
	if (!*strp) return -1;

	return vsprintf_s(*strp, len + 1, fmt, ap);
}
#endif

static SpinLock printfLock;
void threadsafe_printf(const char* args, ...) {
	char* fmtstr = nullptr;
	int msgLen;

	va_list list;
	va_start(list, args);
	msgLen = vasprintf(&fmtstr, args, list);
	va_end(list);

	if (msgLen == -1) return; // allocation error
	if (!fmtstr) return; // if error is passed here

	printfLock.lock();
	printf("%s", fmtstr);
	printfLock.unlock();

	free(fmtstr);
}

static SpinLock fileOutLock;
static FILE* fOut = nullptr;
void threadsafe_outputResult(
	int tsIndex, unsigned int n, double u,
	double Tmin, double Tmax,
	unsigned int M, unsigned int H, double Theta, bool isSMLP,
	double wpiblock, unsigned int dlmiss) {
	fileOutLock.lock();
	if (fOut)
		fprintf(fOut, "%d,%u,%f,%f,%f,%u,%u,%f,%d,%f,%u\n", tsIndex, n, u, Tmin, Tmax, M, H, Theta, isSMLP, wpiblock, dlmiss);
	fileOutLock.unlock();
}

void threadsafe_flushOutput() {
	fileOutLock.lock();
	if (fOut)
		fflush(fOut);
	fileOutLock.unlock();
}

void threadsafe_openOutputFile(const char* outFileName) {
	fileOutLock.lock();
	if (!fOut) {
		fOut = fopen(outFileName, "w");
		if (!fOut)
			return; // probably throw an exception here
		fprintf(fOut,"TS,n,u,Tmin,Tmax,M,H,Theta,SMLP,wpiblock,dlmiss\n");
	}
	fileOutLock.unlock();
}

void threadsafe_closeOutputFile() {
	fileOutLock.lock();
	if (fOut) {
		fclose(fOut);
		fOut = nullptr;
	}
	fileOutLock.unlock();
}