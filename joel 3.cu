
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here


//***********************************************


__global__ void isAlive(int* match, int* isTankAlive, int* dHealth, int T) {
    if (T >= threadIdx.x) {
        int id_ = threadIdx.x;
        if (dHealth[id_] <= 0)
        {
            match++;
            atomicAdd(isTankAlive, -1);
        }
    }
}


__global__ void simulate_round(int M, int N, int T, int round, int* dHealth, int* dHealth_copy, int* dScore, int* dXC, int* dYC) {

    long long idx_source = 1LL * dYC[blockIdx.x] * N + dXC[blockIdx.x];
    long long idx_tar = 1LL * dYC[(blockIdx.x + round) % T] * N + dXC[(blockIdx.x + round) % T];
    long long idx_mid = 1LL * dYC[threadIdx.x] * N + dXC[threadIdx.x];

    __shared__ long long min_dis;
    min_dis = LLONG_MAX;

    int x1 = dXC[blockIdx.x];
    int x2 = dXC[(blockIdx.x + round) % T];
    int x3 = dXC[threadIdx.x];

    int y1 = dYC[blockIdx.x];
    int y2 = dYC[(blockIdx.x + round) % T];
    int y3 = dYC[threadIdx.x];


    bool slope = false;
    if ((1LL * (y1 - y2) * (x1 - x3)) == (1LL * (y1 - y3) * (x1 - x2)))
        slope = true;


    bool common_area = false;
    if ((idx_tar > idx_source) == false && ((idx_mid > idx_source)) == false)
        common_area = true;
    else if ((idx_tar > idx_source) == true && ((idx_mid > idx_source)) == true)
        common_area = true;

    __syncthreads();


    long long middle_dis = abs(idx_source - idx_mid);


    if (blockIdx.x != threadIdx.x)
    {
        if (dHealth[blockIdx.x] > 0)
        {
            if (slope && common_area)
            {
                if (dHealth[threadIdx.x] > 0)
                    atomicMin(&min_dis, middle_dis);
            }
        }
    }__syncthreads();




    if (blockIdx.x != threadIdx.x)
    {
        if (dHealth[blockIdx.x] > 0)
        {
            if (slope && common_area)
            {
                if (dHealth[threadIdx.x] > 0)
                {
                    if (min_dis == middle_dis)
                    {
                        atomicAdd(dHealth_copy + threadIdx.x, -1);
                    }
                    int iid_ = blockIdx.x;
                    if (min_dis == middle_dis) {
                        dScore[iid_]++;
                    }

                }
            }
        }
    }


}


__global__ void init_HP(int* dHealth, int* dHealth_cp, int H, int T) {

    int id = threadIdx.x;
    if (id < T)
    {
        dHealth[threadIdx.x] = H;
    }
    if (T >= id)
    {
        dHealth_cp[id] = H;
    }
}



int main(int argc, char** argv)
{
    // Variable declarations
    int M, N, T, H, * xcoord, * ycoord, * score;


    FILE* inputfilepointer;

    //File Opening for read
    char* inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int*)malloc(T * sizeof(int));  // X coordinate of each tank
    ycoord = (int*)malloc(T * sizeof(int));  // Y coordinate of each tank
    score = (int*)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************


    //create array for health


    int* dScore;
    int* current_alive;
    cudaMalloc(&dScore, (T * sizeof(int)));
    cudaMemset(dScore, 0, T * sizeof(int));

    int* dHealth;
    cudaMalloc(&dHealth, (T * sizeof(int)));

    int* match;
    cudaMalloc(&match, sizeof(int));


    int* dXC;
    cudaMalloc(&dXC, (T * sizeof(int)));
    cudaMemcpy(dXC, xcoord, sizeof(int) * T, cudaMemcpyHostToDevice);

    int* dYC;
    cudaMalloc(&dYC, (sizeof(int) * T));
    cudaMemcpy(dYC, ycoord, sizeof(int) * T, cudaMemcpyHostToDevice);

    int* dHealth_copy;
    current_alive = (int*)malloc(sizeof(int));
    cudaMalloc(&dHealth_copy, (T * sizeof(int)));

    int* dcurrent_alive;
    cudaMalloc(&dcurrent_alive, sizeof(int));


    *current_alive = T;

    init_HP << <1, T >> > (dHealth, dHealth_copy, H, T);

    int round = 1;
    while (1)
    {
        if (*current_alive <= 1)
            break;
        else {
            cudaMemcpy(dcurrent_alive, &T, sizeof(int), cudaMemcpyHostToDevice);
            if (round % T != 0) {
                simulate_round << <T, T >> > (M, N, T, round, dHealth, dHealth_copy, dScore, dXC, dYC);
                int size_ = T * sizeof(int);
                match++;
                cudaMemcpy(dHealth, dHealth_copy, size_, cudaMemcpyDeviceToDevice);
                round++;
                isAlive << <1, T >> > (match, dcurrent_alive, dHealth, T);
                size_ /= T;
                cudaMemcpy(current_alive, dcurrent_alive, size_, cudaMemcpyDeviceToHost);
            }
            else if (round % T == 0)
                round++;
        }
    }



    cudaFree(dHealth);
    cudaFree(dXC);
    cudaMemcpy(score, dScore, sizeof(int) * T, cudaMemcpyDeviceToHost);
    cudaFree(dScore);
    cudaFree(dHealth_copy);
    cudaFree(dYC);


    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char* outputfilename = argv[2];
    char* exectimefilename = argv[3];
    FILE* outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;

}
