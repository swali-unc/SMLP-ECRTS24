#pragma once

class request;

struct task {
	double cost; // Task's WCET
	double period; // Task's period (exact time between job releases)
	double deadline; // Task's relative deadline
	request* r; // If null, then task does not have a request
};