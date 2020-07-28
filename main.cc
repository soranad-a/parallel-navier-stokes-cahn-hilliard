#define _USE_MATH_DEFINES // for M_PI
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include "omp.h"
#include <chrono>
#include "helpers.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
	// constants for the equations
	double eps; // interface thickness epsilon
	double m; // mobility constant m
	double h; // distance between grid points
	double f[2]; // force
	double r; // radius of the bubble
	double eta[2]; // viscosity eta, first value for phi = 1, second for phi = -1
	double rho[2]; // density rho, first value for phi = 1, second for phi = -1
	double sigma; // surface tension sigma

	// constants which are relevant for simulation or runtime
	int num_iterations; // number of iterations
	int print_every_round; // output every x rounds
	double dt; // time step length
	int n_i; // number of grid points in horizontal direction
	int n_j; // number of grid points in vertical direction
	int cnt_threads; // number of threads
	bool output_console; // output in console (true = yes)
	bool output_file; // output in files (true = yes)

	// load constants out of config file
	string configfilename = "config.conf";
	ifstream settings;
	settings.open(configfilename, ios::in);
	cout <<"[*] Try to load variables from file: " << configfilename << "\n";
	if(!settings) {
		printf("[*] Error: File not found!\n");
		settings.close();
		return 0;
	}

	printf("[*] File loaded. Set variables now:\n");
	settings >> eps;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> m;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> h;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> f[0];
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> f[1];
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> r;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> eta[0];
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> eta[1];
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> rho[0];
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> rho[1];
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> num_iterations;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> print_every_round;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> dt;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> n_i;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> n_j;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> cnt_threads;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> sigma;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> output_console;
	settings.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	settings >> output_file;

	settings.close();

	cout<<"eps               = "<<eps<<"\n";
	cout<<"m                 = "<<m<<"\n";
	cout<<"h                 = "<<h<<"\n";
	cout<<"f[2]              = {"<<f[0]<<", "<<f[1]<<"}\n";
	cout<<"r                 = "<<r<<"\n";
	cout<<"eta[2]            = {"<<eta[0]<<", "<<eta[1]<<"}\n";
	cout<<"rho[2]            = {"<<rho[0]<<", "<<rho[1]<<"}\n";
	cout<<"sigma             = "<<sigma<<"\n";
	cout<<"num_iterations    = "<<num_iterations<<"\n";
	cout<<"print_every_round = "<<print_every_round<<"\n";
	cout<<"dt                = "<<dt<<"\n";
	cout<<"n_i               = "<<n_i<<"\n";
	cout<<"n_j               = "<<n_j<<"\n";
	cout<<"cnt_threads       = "<<cnt_threads<<"\n";
	cout<<"output_console    = "<<output_console<<" (1 = True, 0 = False)\n";
	cout<<"output_file       = "<<output_file<<" (1 = True, 0 = False)\n";
	cout<<"\n";

	// Buffer variables
	int cnt = 0;
	string progress_bar = "";
	double percent;

	// benchmark quantities
	double cnt_y_n = 0;
	double cnt_dvy_vy = 0;
	double cnt_dm_y = 0;
	
	// set sigma to correct value
	sigma = sigma * 3 / 2 / sqrt(2);

	// prepare file output
	FILE * fp_benchmark = NULL;
	if (output_file) {
		fp_benchmark = fopen("benchmark.csv", "w");
		fprintf(fp_benchmark, "time; rise velocity; center of mass\n");
	}

	// declaration of all data arrays
	Array2D phi_old(n_i + 2, n_j + 2); // phase field
	Array2D phi_new(n_i + 2, n_j + 2); // phase field
	Array2D mu(n_i + 2, n_j + 2); // mu
	Array2D p_new(n_i + 2, n_j + 2); // pressure
	Array2D p_old(n_i + 2, n_j + 2); // pressure
	Array2D rho_array(n_i + 2, n_j + 2); // density
	Array2D eta_array(n_i + 2, n_j + 2); // viscosity
	Array2D alpha(n_i + 2, n_j + 2);
	Array2D beta(n_i + 2, n_j + 2);

	Array2D gamma(n_i + 3, n_j + 3);

	Array2D u0(n_i + 3, n_j + 2); // velocity in horizontal direction
	Array2D u0_tilde(n_i + 3, n_j + 2); // u0 tilde

	Array2D u1(n_i + 2, n_j + 3); // velocity in vertical direction
	Array2D u1_tilde(n_i + 2, n_j + 3); // u1 tilde

	// prepare vtk output
	VTKWriter writer_phi("phi_", "paraview/");
	VTKWriter writer_u0("u0_", "paraview/");
	VTKWriter writer_u1("u1_", "paraview/");
	VTKWriter writer_p("p_", "paraview/");
	
	omp_set_num_threads(cnt_threads); // number of threads for OpenMP
	high_resolution_clock::time_point t1 = high_resolution_clock::now(); // time measurement

	if (output_console) {
		printf("\r[*] Last print to file no.: - | ETA: infs");
		fflush(stdout);
	}

	// set init condition
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_i + 2; i++) {
		for (int j = 0; j < n_j + 2; j++) {
			phi_old(i, j) = tanh((r - sqrt(((i - 0.5) * h - 0.5) * ((i - 0.5) * h - 0.5) + ((j - 0.5) * h - 0.5) * ((j - 0.5) * h - 0.5))) / (sqrt(2.0) * eps));
			phi_new(i, j) = -1;
		}
	}

	if (output_file) {
		writer_phi.write_step(phi_old, cnt, 0, n_i + 1, 0, n_j + 1, h, -h/2, -h/2);
		writer_phi.finalize();
		writer_u0.write_step(u0, cnt, 0, n_i + 2, 0, n_j + 1, h, -h, -h/2);
		writer_u0.finalize();
		writer_u1.write_step(u1, cnt, 0, n_i + 1, 0, n_j + 2, h, -h/2, -h);
		writer_u1.finalize();
		writer_p.write_step(p_new, cnt, 0, n_i + 1, 0, n_j + 1, h, -h/2, -h/2);
		writer_p.finalize();
	}

	// calculation for num_iterations time steps
	for (int t = 0; t < num_iterations; t++) {
		// reset benchmark quantities every round
		cnt_y_n = 0;
		cnt_dvy_vy = 0;
		cnt_dm_y = 0;
		
		#pragma omp parallel 
		{

			// Navier-Stokes equations
			// calculation of viscosity and density
			#pragma omp for schedule(static)
			for (int i = 0; i < n_i + 2; i++) {
				for (int j = 0; j < n_j + 2; j++) {
					double phi_temp = phi_old(i, j);
					if (phi_temp > 1) phi_temp = 1;
					else if (phi_temp < -1) phi_temp = -1;

					eta_array(i, j) = ((phi_temp + 1) / 2) * (eta[0] - eta[1]) + eta[1];
					rho_array(i, j) = ((phi_temp + 1) / 2) * (rho[0] - rho[1]) + rho[1];
				}
			}

			// helper array gamma
			#pragma omp for schedule(static)
			for (int i = 2; i < n_i + 1; i++) { // only from i=2 to n_i+1 due to free-slip condition at lateral boundaries
				for (int j = 1; j < n_j + 2; j++) {
					gamma(i, j) = 1 / (4 * h) * (u0(i, j) - u0(i, j - 1) + u1(i, j) - u1(i - 1, j))
						* (eta_array(i, j) + eta_array(i - 1, j) + eta_array(i, j - 1) + eta_array(i - 1, j - 1));
				}
			}

			// helper array alpha and beta
			#pragma omp for schedule(static)
			for (int i = 0; i < n_i + 2; i++) {
				for (int j = 0; j < n_j + 2; j++) {
					alpha(i, j) = 2 * eta_array(i, j) * (u0(i + 1, j) - u0(i, j)) / h;
					beta(i, j) = 2 * eta_array(i, j) * (u1(i, j + 1) - u1(i, j)) / h;
				}
			}

			// calculation of temporary velocity
			// u0_tilde
			#pragma omp for schedule(static)
			for (int i = 2; i < n_i + 1; i++) { // only from i=2 to n_i+1 due to vx=0 at lateral boundaries
				for (int j = 1; j < n_j + 1; j++) {
					u0_tilde(i, j) = ((
						(alpha(i, j) - alpha(i - 1, j) + gamma(i, j + 1) - gamma(i, j)) / h // viscosity
						+ ((mu(i, j) + mu(i - 1, j)) * (phi_old(i, j) - phi_old(i - 1, j)) * sigma) / (2 * h * eps) // surface tension
						)
						* 2 / (rho_array(i, j) + rho_array(i - 1, j)) // density
						- (u0(i, j) * (u0(i + 1, j) - u0(i - 1, j)) + 0.25 * (u1(i, j) + u1(i - 1, j)
							+ u1(i - 1, j + 1) + u1(i, j + 1)) * (u0(i, j + 1) - u0(i, j - 1))) / (2 * h) // convection
						)
						* dt + u0(i, j);
				}
			}

			// u1_tilde
			#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) {
				for (int j = 2; j < n_j + 1; j++) {  // only from j=2 to n_j+1 due to vy=0 at lateral boundaries
					u1_tilde(i, j) = ((
						(beta(i, j) - beta(i, j - 1) + gamma(i + 1, j) - gamma(i, j)) / h // viscosity
						+ f[1] / 2 * (rho_array(i, j) + rho_array(i, j - 1) - 2.0 * rho[1]) // force
						+ ((mu(i, j) + mu(i, j - 1)) * (phi_old(i, j) - phi_old(i, j - 1)) * sigma) / (2 * h * eps) // surface tension
						)
						* 2 / (rho_array(i, j) + rho_array(i, j - 1)) // density
						- (0.25 * (u0(i, j) + u0(i, j - 1) + u0(i + 1, j) + u0(i + 1, j - 1)) * (u1(i + 1, j) - u1(i - 1, j))
							+ u1(i, j) * (u1(i, j + 1) - u1(i, j - 1))) / (2 * h) // convection
						)
						* dt + u1(i, j);
				}
			}

			// calculation of pressure p
			int iter = 10;
			for (int count=0; count<iter; count++) { // solve iter time steps of pressure calculation 
				
				#pragma omp for schedule(static)
				for (int i = 1; i < n_i + 1; i++) {
					for (int j = 1; j < n_j + 1; j++) {
						p_new(i, j) = (
							0.125 * (p_old(i + 1, j) + p_old(i - 1, j) + p_old(i, j + 1) + p_old(i, j - 1) - 4 * p_old(i, j))
							+ rho_array(i, j) * 0.125 / 4 * ((1 / rho_array(i + 1, j) - 1 / rho_array(i - 1, j)) * (p_old(i + 1, j) - p_old(i - 1, j))
								+ (1 / rho_array(i, j + 1) - 1 / rho_array(i, j - 1)) * (p_old(i, j + 1) - p_old(i, j - 1)))
							- h * rho_array(i,j) * 0.125 / dt * (u0_tilde(i + 1, j) - u0_tilde(i, j) + u1_tilde(i, j + 1) - u1_tilde(i, j))
							)
							+ p_old(i, j);
					}
				}

				#pragma omp single
				{
					swap(p_old, p_new);
				}
			}
			#pragma omp barrier
			#pragma omp single 
			{
				swap(p_old, p_new); // finally swap p_old and p_new again to ensure p_new is the last computed value
			}

			// calculation of new velocities
			// u0
			#pragma omp for schedule(static)
			for (int i = 2; i < n_i + 1; i++) { // only from i=2 to n_i+1 due to vx=0 at lateral boundaries
				for (int j = 1; j < n_j + 1; j++) {
					u0(i, j) = (-(p_new(i, j) - p_new(i - 1, j)) / h
						* 2 / (rho_array(i, j) + rho_array(i - 1, j)))
						* dt + u0_tilde(i, j);
				}
			}

			// u1
			#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) { // only from j=2 to n_j+1 due to vy=0 at lateral boundaries
				for (int j = 2; j < n_j + 1; j++) {
					u1(i, j) = (-(p_new(i, j) - p_new(i, j - 1)) / h
						* 2 / (rho_array(i, j) + rho_array(i, j - 1)))
						* dt + u1_tilde(i, j);
				}
			}

			// Cahn-Hilliard equation
			// calculation of mu
			#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) {
				for (int j = 1; j < n_j + 1; j++) {
					mu(i,j) = (-(eps * eps) / (h * h))
						* (phi_old(i + 1, j) + phi_old(i - 1, j) + phi_old(i, j + 1) + phi_old(i, j - 1) - 4 * phi_old(i, j))
						+ (phi_old(i, j) * phi_old(i, j) * phi_old(i, j)) - phi_old(i, j);
				}
			}

			// calculation of phase field phi
			#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) {
				for (int j = 1; j < n_j + 1; j++) {
					phi_new(i, j) = ((m / (h * h))
						* (mu(i + 1, j) + mu(i - 1, j) + mu(i, j + 1) + mu(i, j - 1) - 4 * mu(i, j))
						- ((( u0(i, j) + u0(i + 1, j)) / 2) * (phi_old(i + 1, j) - phi_old(i - 1, j))
						+ ((u1(i, j) + u1(i, j + 1)) / 2) * (phi_old(i, j + 1) - phi_old(i, j - 1))) / (2 * h))
						* dt + phi_old(i, j);
				}
			}

			// reset boundaries to correct boundary conditions
			#pragma omp for schedule(static)
			for (int i = 0; i < n_i + 2; i++) {
				p_new(i, 0) = p_new(i, 1);
				p_new(i, n_j + 1) = p_new(i, n_j);				
				phi_new(i, 0) = phi_new(i, 1);
				phi_new(i, n_j + 1) = phi_new(i, n_j);
				u0(i,0) = - u0(i,1);
				u0(i,n_j+1) = - u0(i,n_j);
			}
			#pragma omp for schedule(static)
			for (int j = 0; j < n_j + 2; j++) {
				p_new(0, j) = p_new(1, j);
				p_new(n_i + 1, j) = p_new(n_i, j);
				phi_new(0, j) = phi_new(1, j);
				phi_new(n_i + 1, j) = phi_new(n_i, j);		
				u1(0,j) = u1(1,j);
				u1(n_i+1,j) = u1(n_i,j);
			}
		}
		
		// swap fields of pressure and phase
		swap(p_old, p_new);
		swap(phi_new, phi_old);
		
		// output
		if (output_file)
			if (t % print_every_round == 0 || t == num_iterations - 1) {
				cnt++;
				writer_phi.write_step(phi_old, cnt, 0, n_i + 1, 0, n_j + 1, h, -h/2, -h/2);
				writer_phi.finalize();
				writer_u0.write_step(u0, cnt, 0, n_i + 2, 0, n_j + 1, h, -h, -h/2);
				writer_u0.finalize();
				writer_u1.write_step(u1, cnt, 0, n_i + 1, 0, n_j + 2, h, -h/2, -h);
				writer_u1.finalize();
				writer_p.write_step(p_old, cnt, 0, n_i + 1, 0, n_j + 1, h, -h/2, -h/2);
				writer_p.finalize();

				// calculate benchmark quantities
				#pragma omp parallel for schedule(static) reduction(+:cnt_dvy_vy, cnt_dm_y, cnt_y_n)
				for(int i = 0; i < n_i + 2; i++) {
					for (int j = 0; j < n_j + 2; j++) {

						cnt_dvy_vy += (phi_old(i, j) + 1) * (u1(i, j) + u1(i, j + 1)) / 2;
						cnt_dm_y += (phi_old(i, j) + 1) * (j - 0.5);
						cnt_y_n += phi_old(i, j) + 1;
					}
				}
				fprintf (fp_benchmark, "%f; %f; %f\n",(t + 1) * dt, cnt_dvy_vy / cnt_y_n, h * cnt_dm_y / cnt_y_n);
			
				if (output_console) {
					// print progress
					high_resolution_clock::time_point t2 = high_resolution_clock::now();
					duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
					printf("\r[*] Last print to file no.: %i | ETA: %5.fs", cnt, (time_span.count() / cnt) * (num_iterations / print_every_round) - time_span.count());
					fflush(stdout);
				}
			}
	}

	if (output_console) {
		high_resolution_clock::time_point t3 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t3 - t1);
		printf("\nTime needed: %f\n", time_span.count());
	}

	if (output_file) {
		fclose (fp_benchmark);
	}

	return 0;
}
