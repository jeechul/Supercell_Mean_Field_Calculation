#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <mpi.h>
#include "../include/Lattice_geometry.h"
#include "../include/Eigen_system.h"
#include "../include/Equations.h"
#include "../include/argparse.h"

#define __OUTFILE__

using LATTICE = KAGOME;

std::vector<std::vector<int>> make_lattice(int l);

struct PROFILE
{
    const static int L=24;
    const static int L2 = L*2;
    const static int N_tot = 32;
    constexpr static double RBZ_x=1./2.,RBZ_y=1./2.; // For unit cell : RBZ_x=2, RBZ_y=2
    static std::vector<std::vector<int>> lattices;
    static double V;
};

// 12 sites //
/* std::vector<std::vector<int>> PROFILE::lattices = {{1,0},{3,1},{4,3},{0,4},{7,6},{9,7},{10,9},{6,10},// k_x
        {2,0},{6,2},{8,6},{0,8},{5,3},{9,5},{11,9},{3,11},// k_y
        {1,2},{11,1},{7,8},{5,7},{4,5},{8,4},{10,11},{2,10}}; //k_xy
double PROFILE::V = 8.*sqrt(3);
*/

// 18 sites //
/* std::vector<std::vector<int>> PROFILE::lattices = {{1,0},{8,1},{9,8},{15,9},{17,15},{0,17},{4,3},{11,4},{6,11},{12,6},{14,12},{3,14},// k_x
        {2,3},{8,2},{7,8},{12,7},{13,12},{0,13},{5,0},{11,5},{10,11},{15,10},{16,15},{3,16},// k_y
        {2,1},{4,2},{5,4},{1,5},{7,6},{9,7},{10,9},{6,10},{14,13},{16,14},{17,16},{13,17}}; //k_xy
double PROFILE::V = 2.*sqrt((sqrt(13)+sqrt(7)+2)*(sqrt(13)+sqrt(7)-2)*(sqrt(13)-sqrt(7)+2)*(-sqrt(13)+sqrt(7)+2));
*/

// 24 sites //
std::vector<std::vector<int>> PROFILE::lattices = {{1,0},{6,1},{8,6},{15,8},{16,15},{21,16},{23,21},{0,23},{4,3},{9,4},{11,9},{12,11},{13,12},{18,13},{20,18},{3,20},// k_x
        {2,3},{6,2},{7,6},{12,7},{5,0},{9,5},{10,9},{15,10},{14,15},{18,14},{19,18},{0,19},{17,12},{21,17},{22,21},{3,22},// k_y
        {2,1},{4,2},{5,4},{1,5},{8,7},{10,8},{11,10},{7,11},{14,13},{16,14},{17,16},{13,17},{20,19},{22,20},{23,22},{19,23}}; // k_xy
double PROFILE::V = 16.*sqrt(3);


// 30 sites //
/* std::vector<std::vector<int>> PROFILE::lattices = {{4,3},{9,4},{10,9},{1,0},{6,1},{7,6},{15,7},{16,15},{12,10},{13,12},{21,13},{22,21},{24,22},{18,16},{19,18},{27,19},{28,27},{0,28},{25,24},{3,25}, // k_x
        {5,3},{6,5},{8,6},{12,8},{14,12},{18,14},{20,18},{24,20},{26,24},{0,26},{2,0},{9,2},{11,9},{15,11},{17,15},{21,17},{23,21},{27,23},{29,27},{3,29}, // k_y
        {1,2},{5,1},{4,5},{2,4},{8,10},{7,8},{11,7},{10,11},{14,16},{13,14},{17,13},{16,17},{19,20},{23,19},{22,23},{20,22},{25,26},{29,25},{28,29},{26,28}}; // k_xy
double PROFILE::V = 2.*sqrt((sqrt(21)+sqrt(31)+2)*(sqrt(21)+sqrt(31)-2)*(sqrt(21)-sqrt(31)+2)*(-sqrt(21)+sqrt(31)+2));	
*/

// large system //
//std::vector<std::vector<int>> PROFILE::lattices = make_lattice(4);
//double PROFILE::V = 32*sqrt(3); // 3*4*4 sites

using EIGEN = EigenSetRandomPot<LATTICE,PROFILE>;
using EQUATION = Gauss<EIGEN,PROFILE>;

int main(int argc, char* argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction strength"));
    options.push_back(pair_t("X", "disorder strength"));
    options.push_back(pair_t("path", "directory to load and save files"));
    options.push_back(pair_t("xpath", "directory to load parameter's files"));
    // env; default value
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    const int L = PROFILE::L, 
        L2 = PROFILE::L2,
        N_tot = PROFILE::N_tot;
    const double U = parser.find<double>("U"),
        X = parser.find<double>("X"),
        beta = 200,
        V = PROFILE::V;
    std::vector<double> m(L), d(L);

    int ranks, nprocs;

	MPIERR(MPI_Init(&argc,&argv)); 
	MPIERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
	MPIERR(MPI_Comm_rank(MPI_COMM_WORLD, &ranks));

	std::string filename, lfilename;

    double gamma[L];
    std::fill(gamma,gamma+L,1.0);

    EIGEN eigen(L,gamma);
    EQUATION equation(eigen,L,63);

    filename = parser.find<>("path") + "/RandomUniPotKagome_SFW-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + std::to_string(N_tot) + ".dat"; 
#ifdef __OUTFILE__
    std::ofstream outfile;
    MPIERR(MPI_Barrier(MPI_COMM_WORLD));
    if (!std::ifstream(filename).is_open())
    {
        outfile.open(filename);
        if (ranks == 0)
            outfile << "#         D^s_conv       D^s_geom        D^s        " << std::endl;
    }
    else
        outfile.open(filename);
    outfile.precision(10);
    MPIERR(MPI_Barrier(MPI_COMM_WORLD));
#endif
    lfilename = parser.find<>("xpath") + "/KagomeProfile-L" + std::to_string(L) + "U" + parser.find<>("U") + "X" + parser.find<>("X") + "N" + std::to_string(N_tot) + ".dat";
    if (!std::ifstream(lfilename).is_open())
    {
        if (ranks == 0) std::cerr << "There is not the file!" << std::endl;
        exit(1);
    }
    else 
    {
        int i = 0;
        double Ntot, mg;
        std::ifstream loadfile(lfilename);
        do
        {
            std::string profile;
            std::getline(loadfile,profile);
            std::istringstream iss(profile);

            if (profile[0] != '#')
            {
                for (int j=0;j<L;++j){
                    iss >> gamma[j] >> d[j] >> m[j];
                }
                iss >> Ntot >> mg;
            }
            else
                continue;

            eigen.set_xconf(gamma);
            std::pair<double,double> sfw = equation.SFW(m,d,V,beta,ranks,nprocs);
#ifdef __OUTFILE__		
            if(ranks == 0)
            {
                outfile << "\t" << sfw.first << "\t" << sfw.second << "\t" << sfw.first+sfw.second << std::endl;
                std::cout << "\r# --- Estimating ... " << std::setw(7) << i+1 << std::flush; 
            }
#else
            if(ranks == 0)
            {
                std::cout << "\t" << sfw.first << "\t" << sfw.second << "\t" << sfw.first+sfw.second << std::endl;
            }
#endif
            i++;
        } while (!loadfile.eof());
        if (ranks == 0) std::cout << std::endl;
    }
#ifdef __OUTFILE__
    outfile.close();
#endif
    MPIERR(MPI_Finalize());

    return 0;
}

std::vector<std::vector<int>> make_lattice(int l)
{
    std::vector<std::vector<int>> lattices;

    //cout<<"--------kx-----------"<<endl;
    for(int cell=0;cell<l*l;cell++){
        std::vector<int> s1,s2;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int right = 3*((cX+1)%l+l*(cY%l));

        s1 = {1+center,center};
        s2 = {right,1+center};

        lattices.push_back(s1);
        lattices.push_back(s2);
        //cout<<"{"<<s1[0]<<","<<s1[1]<<"}"<<endl;
        //cout<<"{"<<s2[0]<<","<<s2[1]<<"}"<<endl;
    }
    //cout<<endl;

    //cout<<"---------ky-----------"<<endl;
    for(int cell=0;cell<l*l;cell++){
        std::vector<int> s1,s2;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int up = 3*(cX%l+l*((cY+1)%l));

        s1 = {2+center,center};
        s2 = {up,2+center};

        lattices.push_back(s1);
        lattices.push_back(s2);
        //cout<<"{"<<s1[0]<<","<<s1[1]<<"}"<<endl;
        //cout<<"{"<<s2[0]<<","<<s2[1]<<"}"<<endl;
    }
    //cout<<endl;

    //cout<<"-----------kxy-----------"<<endl;
    for(int cell=0;cell<l*l;cell++){
        std::vector<int> s1,s2;
        int cX = cell%l;
        int cY = cell/l;

        int center = 3*cell;
        int cross = 3*((cX+l-1)%l+l*((cY+1)%l));

        s1 = {1+center,2+center};
        s2 = {2+center,1+cross};

        lattices.push_back(s1);
        lattices.push_back(s2);
        //cout<<"{"<<s1[0]<<","<<s1[1]<<"}"<<endl;
        //cout<<"{"<<s2[0]<<","<<s2[1]<<"}"<<endl;
    }

    return lattices;
}


