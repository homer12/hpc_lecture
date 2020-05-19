#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  /*
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      if(i != j) {
        double rx = x[i] - x[j];
        double ry = y[i] - y[j];
        double r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
  */
	
	__m256 xvec = _mm256_load_ps(x);
	__m256 yvec = _mm256_load_ps(y);
	__m256 mvec = _mm256_load_ps(m);
	for( int i = 0; i < N; i++ ){
		__m256 rxvec = _mm256_set1_ps(x[i]);
		__m256 ryvec = _mm256_set1_ps(y[i]);
		__m256 fxvec = _mm256_setzero_ps();
		__m256 fyvec = _mm256_setzero_ps();
	
		__m256 rreci, r3reci, rxxvec, ryyvec, noise;
		__m256 dfx, dfy, tmp;
		
		// x[i]-x[0], x[i]-x[1], ..., x[i]-x[7]
		rxvec = _mm256_sub_ps( rxvec, xvec );
		ryvec = _mm256_sub_ps( ryvec, yvec );
		
		// (x[i]-x[j])^2
		rxxvec = _mm256_mul_ps( rxvec, rxvec );
		ryyvec = _mm256_mul_ps( ryvec, ryvec );
		
		// 1 / ( rx^2 + ry^2 + 10e-20 )
		// why use 10e-20:
		// It's just a random number so that
		// when computing reciprocal it will
		// not cause 1.0/0.0, so I don't have
		// to use any kind of masking.
		//
		// I think 10e-20 is small enough
		// to not make much difference to
		// the final result.
		//
		// In fact, the result is somewhat different
		// from the original.
		// But I don't think that's by cause of
		// the constant I added.
		rreci = _mm256_add_ps( rxxvec, ryyvec );
		noise = _mm256_set1_ps( 10e-20 );
		rreci = _mm256_add_ps( rreci, noise );
		rreci = _mm256_rsqrt_ps( rreci );
		
		// 1/r^3
		r3reci = _mm256_mul_ps( rreci, rreci );
		r3reci = _mm256_mul_ps( r3reci, rreci );

		// 1.0 * rx * m[j] / r^3
		dfx = _mm256_set1_ps(1.0);
		dfx = _mm256_mul_ps( dfx, rxvec );
		dfx = _mm256_mul_ps( dfx, mvec );
		dfx = _mm256_mul_ps( dfx, r3reci );
		
		// 1.0 * ry * m[j] / r^3
		dfy = _mm256_set1_ps(1.0);
		dfy = _mm256_mul_ps( dfy, ryvec );
		dfy = _mm256_mul_ps( dfy, mvec );
		dfy = _mm256_mul_ps( dfy, r3reci );
		
		// 0 - dfx, 0 - dfy
		fxvec = _mm256_sub_ps( fxvec, dfx );
		fyvec = _mm256_sub_ps( fyvec, dfy );
		
		// do reduction
		// slide P18 from 0514.pdf
		tmp = _mm256_permute2f128_ps( fxvec, fxvec, 1 );
		tmp = _mm256_add_ps( tmp, fxvec );
		tmp = _mm256_hadd_ps( tmp, tmp );
		tmp = _mm256_hadd_ps( tmp, tmp );
		_mm256_store_ps( fx, tmp );
		
		
		tmp = _mm256_permute2f128_ps( fyvec, fyvec, 1 );
		tmp = _mm256_add_ps( tmp, fyvec );
		tmp = _mm256_hadd_ps( tmp, tmp );
		tmp = _mm256_hadd_ps( tmp, tmp );
		_mm256_store_ps( fy, tmp );
		
		printf("%d %g %g\n", i, fx[0], fy[0]);

		/*float tmp[8];
		_mm256_store_ps( tmp, rxvec );
		printf("%d:", i);
		for(int j = 0; j < N; j++ )	printf(" %.3f", tmp[j]);
		printf("\n");*/
	}

}
