#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>


__global__ void bucketSort( int *key, int keylen, int *bucket, int range ){
	int rank = blockIdx.x * blockDim.x + threadIdx.x;
	
	// I found that if the numbers of threads in each block are not equal
	// there are some bugs.
	// So it's better to filter high rank process to avoid to
	// exceed the upper limit.
	if( rank >= keylen )
		return;
	
	// Initialize bucket[] in global memory
	if( rank < range )	
		bucket[rank] = 0;
	__syncthreads();

	
	// Update bucket[] atomically
	atomicAdd( bucket+key[rank], 1 );
	__syncthreads();
	
	// FOR DEBUG
	/* 
	if( rank == 0 ){
		printf("\n");
		for( int i = 0; i < range; i++ )
			printf("bucket[%d] = %d\n", i, bucket[i]);
		printf("\n");
	}
	*/

	// Copy data from bucket[] to key[rank]
	// for j = 0 ~ b[0]-1, key[j] = 0
	// for j = b[0] ~  b[0]+b[1]-1, key[j] = 1
	// for j = b[0]+b[1] ~ b[0]+b[1]+b[2]-1, key[j] = 2
	// etc
	int bIdx = -1, bItems = 0;
	do{
		bItems += bucket[ ++bIdx ]; 
	}while( bItems <= rank );
	key[rank] = bIdx;
}


int main() {
  const int n = 50;
  const int m = 20;
  const int range = 5;
  
  //std::vector<int> key(n);
  int *key;
  cudaMallocManaged( &key, n*sizeof(int) );
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
  }

  //std::vector<int> bucket(range); 
  int *bucket;
  cudaMallocManaged( &bucket, range*sizeof(int) );
  

  // `key` and `bucket` will reside in global memory
  clock_t before = clock();
  bucketSort<<< (n+m-1)/m, m >>>( key, n, bucket, range );
  clock_t after = clock();
 	
  
  cudaDeviceSynchronize();
}
