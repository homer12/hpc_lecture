#include <cstdio>
#include <cstdlib>
#include <vector>


__global__ void bucketSort( int *key, int *bucket, int range ){
	int rank = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	// Initialize bucket[] in global memory
	if( rank < range )	
		bucket[rank] = 0;
	__syncthreads();

	
	// Update bucket[] atomically
	atomicAdd( bucket+key[rank], 1 );
	__syncthreads();
	
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
  const int m = 25;
  const int range = 5;
  
  //std::vector<int> key(n);
  int *key;
  cudaMallocManaged( &key, n*sizeof(int) );
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  //std::vector<int> bucket(range); 
  int *bucket;
  cudaMallocManaged( &bucket, range*sizeof(int) );
  

  // `key` and `bucket` will reside in global memory
  bucketSort<<< (n+m-1)/m, m >>>( key, bucket, range );
 	

  /*
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  */
  
  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
