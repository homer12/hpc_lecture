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

void pressurePoission( matrix &p, float dx, float dy, matrix &b){
	
}
