#include <cstdio>
#include <mpi.h>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

typedef vector<vector<float>> matrix;

int irank, nsize;

void buildUpB( matrix &b, float rho, float dt, matrix &u, matrix &v, float dx, float dy){
	int row = b.size();
	int col = b[0].size();

	// parallize row loop
	int begin = (row / nsize) * irank;
	int end = (row / nsize) * (irank + 1);


	for( int j = begin; j < end; j++ ){
		if( j == 0 || j == row-1 )
			continue;

		for( int i = 1; i < col-1; i++ ){
			float dudx = ( u[j][i+1] - u[j][i-1] ) / (2.0 * dx);
			float dudy = ( u[j+1][i] - u[j-1][i] ) / (2.0 * dy);
			float dvdx = ( v[j][i+1] - v[j][i-1] ) / (2.0 * dx);
			float dvdy = ( v[j+1][i] - v[j-1][i] ) / (2.0 * dy);

			b[j][i] = 1 / dt * (dudx+ dvdy);
			b[j][i] -= dudx * dudx;
			b[j][i] -= 2 * dudy * dvdx;
			b[j][i] -= dvdy * dvdy;
		}
	}

	matrix recvb( row, vector<float>(col));
	MPI_Allgather( &b[begin][0], (end-begin)*col, MPI_FLOAT,
					&recvb[0][0], (end-begin)*col, MPI_FLOAT,
					MPI_COMM_WORLD);
	
	b = recvb;
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
			p[j][col-1] = p[j][col-2];
			p[j][0] = p[j][1];
		}

		// dp/dy = 0 at y = 0
		// p = 0 at y = 2
		for( int i = 0; i < col; i++ ){
			p[0][i] = p[1][i];
			p[row-1][i] = 0;
		}
	}
}

void cavity_flow( int nt,
	matrix &u, matrix &v,
	float dt, float dx, float dy,
	matrix &p, float rho, float nu, int nit){
	
	
	int row = u.size();
	int col = u[0].size();
	
	matrix b(row, vector<float>(col, 0));
	float tmp;

	for( int t = 0; t < nt; t++ ){
		matrix un = u;
		matrix vn = v;

		// buildUpB( matrix &b, float rho, float dt, matrix &u, matrix &v, float dx, float dy)
		buildUpB( b, rho, dt, u, v, dx, dy );
		// pressurePoission( matrix &p, float rho, float dx, float dy, matrix &b, int nit)
		pressurePoission( p, rho, dx, dy, b, nit);

		for( int j = 1; j < row-1; j++ ){
			for( int i = 1; i < col-1; i++ ){
				u[j][i] = un[j][i];
				u[j][i] -= un[j][i] * dt / dx * (un[j][i] - un[j][i-1]);
				u[j][i] -= v[j][i] * dt / dy * (un[j][i] - un[j-1][i]);
				u[j][i] -= dt / (rho * 2 * dx) * (p[j][i+1] - p[j][i-1]);
				tmp = dt / (dx*dx) * (un[j][i+1] - 2*un[j][i] + un[j][i-1]);
				tmp += dt / (dy*dy) * (un[j+1][i] - 2*un[j][i] + un[j-1][i]);
				u[j][i] += nu * tmp;

				v[j][i] = vn[j][i];
				v[j][i] -= un[j][i] * dt / dx * (vn[j][i] - vn[j-1][i]);
				v[j][i] -= dt / (rho * 2 * dy) * (p[j+1][i] - p[j-1][i]);
				tmp = dt / (dx*dx) * (vn[j][i+1] - 2*vn[j][i] + vn[j][i-1]);
				tmp += dt / (dy*dy) * (vn[j+1][i] - 2*vn[j][i] + vn[j-1][i]);
				v[j][i] += nu * tmp;
			}
		}


		// set u[0,:] = u[-1,:] = v[0,:] = v[-1,:] = 0
		for( int i = 0; i < col; i++ ){
			u[0][i] = 0;
			u[row-1][i] = 0;
			v[0][i] = 0;
			v[row-1][i] = 0;
		}

		// set u[:,0] = u[:,-1] = v[:,0] = v[:,-1] = 0
		for( int j = 0; j < row; j++ ){
			u[j][0] = 0;
			u[j][col-1] = 0;
			v[j][0] = 0;
			v[j][col-1] = 0;
		}
	}
}



int main( int argc, char *argv[] ){
	MPI_Init( &argc, &argv );
	MPI_Comm_size( MPI_COMM_WORLD, &nsize );
	MPI_Comm_rank( MPI_COMM_WORLD, &irank );


	int nx = 40;
	int ny = 40;
	int nt = 100;
	int nit = 50;
	int c = 1;
	float dx = 2.0 / (nx - 1);
	float dy = 2.0 / (ny - 1);

	float rho = 1;
	float nu = 0.1;
	float dt = 0.001;

	matrix u( ny, vector<float>(nx, 0) );
	matrix v( ny, vector<float>(nx, 0) );
	matrix p( ny, vector<float>(nx, 0) );
	//matrix b( ny, vector<float>(nx, 0) );

	/*
	cavity_flow( int nt,
		matrix &u, matrix &v,
		float dt, float dx, float dy,
		matrix &p, float rho, float nu, int nit);
	*/
	cavity_flow( nt, u, v, dt, dx, dy,
		p, rho, nu, nit);

	return 0;
}
