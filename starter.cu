/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__constant__ long int cfilter[2401];

__global__ void PerformCudaConvolutionOperation(long int*gmat,int m,int n,int k,long int*gans)
{
     __shared__ int s[5000];
     __shared__ int sr;
     __shared__ int er;
     __shared__ int sr2;
     __shared__ int nrb;
    
    long int id=blockIdx.x * blockDim.x+threadIdx.x;
    long int r=id/n;
    long int c=id%n;
    if(threadIdx.x==0)
    {
        sr=(r-k/2)<0?0:(r-k/2);
    }
    else if(threadIdx.x==1023)
    {
        er=(r+k/2)>m-1?m-1:(r+k/2);
    }
    else if(id==(m*n)-1)
      er=m-1;
    __syncthreads();
    nrb=er-sr+1;
    sr2=sr;
    if(threadIdx.x==0)
    {
        for(int i=0;sr<=er;i++,sr++)
        {
            for(int j=0;j<n;j++)
            s[i*n+j]=gmat[sr*n+j];
        }
    }
    __syncthreads();
    long int i,j,x,y;
     if(r==0)
     {
         x=k/2;
         i=r;
        if(c==0)
        {
             y=k/2;
             j=c;
        }
        else
        {
            y=(k/2-c)<0?0:(k/2-c);
            j=(c-k/2)<0?0:(c-k/2);
        }
    }
    else if(r>0)
    {
       x=(k/2-r)<0?0:(k/2-r);
       i=(r-k/2)<0?0:(r-k/2);
       if(c==0)
       {
          y=k/2;
          j=c;
       }
       
       else
       {
        j=(c-k/2)<0?0:(c-k/2);
        y=(k/2-c)<0?0:(k/2-c);
       }

    }
    i=i-sr2;
    long int sj=j;
    for(int p=x;p<k&&i<nrb;p++,i++)
    {
        for(int q=y;q<k&&j<n;q++,j++)
        gans[r*n+c]+= cfilter[p*k + q] * s[i * n + j];
        j=sj;

    }
    
}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    long int* gmat;
    long int* gans;
    cudaMemcpyToSymbol(cfilter, h_filter, k * k * sizeof(long int), 0, cudaMemcpyHostToDevice);
    cudaMalloc(&gmat,m*n*sizeof(long int));
    cudaMemcpy(gmat,h_mat,m*n*sizeof(long int),cudaMemcpyHostToDevice);
    cudaMalloc(&gans,m*n*sizeof(long int));
    int blocks=ceil((m*n)/1024.0);

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    PerformCudaConvolutionOperation<<<blocks,1024>>>(gmat,m,n,k,gans);
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    cudaMemcpy(h_ans,gans,m*n*sizeof(long int),cudaMemcpyDeviceToHost);
    cudaFree(gmat);
    cudaFree(gans);
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}