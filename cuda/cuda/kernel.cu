
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iterator>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdint.h>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void printToFile(int* arr, int n) {
	ofstream fstream;
	fstream.open("data/output");
	for (int i = 0; i < n; i++) {
		fstream << arr[i] << endl;
	}
}

void printArr(int* arr, int n) {
	for (int i = 0; i < n; i++) {
		cout << arr[i] << endl;
	}
}

void rng(int* arr, int n) {
	int seed = 13516154;
	srand(seed);
	for (long i = 0; i < n; i++) {
		arr[i] = (int)rand();
	}
}

// parallel radix sort
// get specific bit at index = idx
__global__ void generateFlag(int* flag, int* arr, int n, int idx) {

	// parallel
	for (int i = 0; i < n; i++) {
		if ((arr[i] >> idx) & 1 == 1) {
			flag[i] = 1;
		}
		else {
			flag[i] = 0;
		}
	}
}

// create I-down array
int* generateIDown(int* flag, int n) {
	int* iDown = (int*)malloc(n * sizeof(int));
	int val = 0;

	iDown[0] = val;
	for (int i = 1; i < n; i++) {
		if (flag[i - 1] == 0) {
			val++;
		}
		iDown[i] = val;
	}
	return iDown;
}

// create I-up array
int* generateIUp(int* flag, int n) {
	int* iUp = (int*)malloc(n * sizeof(int));
	int val = n - 1;

	iUp[n - 1] = val;
	for (int i = n - 2; i >= 0; i--) {
		if (flag[i + 1] == 1) {
			val--;
		}
		iUp[i] = val;
	}
	return iUp;
}

int* generateShouldIndex(int* flag, int* iDown, int* iUp, int n) {
	int* shouldIndex = (int*)malloc(n * sizeof(int));

	// parallel
	for (int i = 0; i < n; i++) {
		if (flag[i] == 0) {
			shouldIndex[i] = iDown[i];
		}
		else {
			shouldIndex[i] = iUp[i];
		}
	}
	return shouldIndex;
}

void permute(int* arr, int* flag, int* iDown, int* iUp, int n) {
	int* shouldArr = (int*)malloc(n * sizeof(int));

	int* shouldIndex = generateShouldIndex(flag, iDown, iUp, n);

	// parallel
	for (int i = 0; i < n; i++) {
		shouldArr[shouldIndex[i]] = arr[i];
	}

	// parallel
	for (int i = 0; i < n; i++) {
		arr[i] = shouldArr[i];
	}
}

void split(int* arr, int n, int idx) {
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

  int* h_flag = (int*)malloc(n * sizeof(int));
  int* d_flag;

  int* d_arr;

  cudaMalloc(&d_flag, n * sizeof(int));
  cudaMalloc(&d_arr, n * sizeof(int));

  cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

	generateFlag<<<numBlocks,blockSize>>>(d_flag, d_arr, n, idx);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_flag, d_flag, n * sizeof(int), cudaMemcpyDeviceToHost);

	int* iDown = generateIDown(h_flag, n);
	int* iUp = generateIUp(h_flag, n);

  permute(arr, h_flag, iDown, iUp, n);
  cout << "progress: ";
  printArr(arr, n);
}


void radixSort(int* arr, int n) {
	int idx = 0;

	for (int i = 0; i < 32; i++) {
		split(arr, n, i);
	}
}

int main(int argc, char** argv)
{
	int n = 2;

	//if (argc != 2) {
		//cout << "Wrong input" << endl;
		//cout << "./radix_sort <N>" << endl;
		//exit(0);
	//}
	//else {
		//n = int(argv[1]);
	//}


	int * arr = (int*)malloc(n * sizeof(int));

	rng(arr, n);
	printArr(arr,n);

	clock_t beginTime = clock();
  radixSort(arr, n);
  clock_t endTime = clock();	
  printArr(arr, n);

	double elapsedTime = (double)endTime - beginTime / CLOCKS_PER_SEC;

	cout << "Parallel Radix Sort Time: " << elapsedTime << endl;
	cout << endl;

    return 0;
}
