#include <cstdio>
#include <vector>
using namespace std;

typedef vector<vector<float>> matrix;


void buildUpB( matrix &b, float rho, float dt, matrix &u, matrix &v, float dx, float dy){
	int row = b.size();
	int col = b[0].size();

	for( int j = 1; j < row-1; j++ ){
		for( int i = 1; i < col-1; i++ ){
			float dudx = ( u[j][i+1] - u[j][i-1] ) / (2.0 * dx);
			float dudy = ( u[j+1][i] - u[j-1][i] ) / (2.0 * dy);
			float dvdx = ( v[j][i+1] - v[j][i-1] ) / (2.0 * dx);
			float dvdy = ( v[j+1][i] - v[j-1][i] ) / (2.0 * dy);

			d[j][i] = 1 / dt * (dudx+ dvdy);
			d[j][i] -= dudx * dudx;
			d[j][i] -= 2 * dudy * dvdx;
			d[j][i] -= dvdy * dvdy;
		}
	}
}

void pressurePoission( matrix &p, float rho, float dx, float dy, matrix &b, int nit){
	int row = p.size();
	int col = p[0].size();

	for( int k = 0; k < nit; k++ ){
		matrix pn = p;

		for( int j = 1; j < row-1; j++ ){
			for( int i = 1; i < col-1; i++ ){
				float tmp = ( pn[j][i+1] + p[j][i-1] ) * dy * dy;
				tmp += ( pn[j+1][i] + pn[j-1][i] ) * dx * dx;
				tmp /= 2 * ( dx*dx + dy*dy );
				p[j][i] = tmp;

				tmp = rho * dx*dx * dy*dy / (2 * (dx*dx + dy*dy));
				tmp *= b[j][i];
				p[j][i] -= tmp;
			}
		}
		
		// dp/dx = 0 at x = 2 and x = 0
		for( int j = 0; j < row; j++ ){
			dp[j][col-1] = dp[j][col-2];
			dp[j][0] = dp[j][1];
		}

		// dp/dy = 0 at y = 0
		// p = 0 at y = 2
		for( int i = 0; i < col; i++ ){
			dp[0][i] = dp[1][i];
			dp[row-1][i] = 0;
		}
	}
}
