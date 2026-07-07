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
#include "../include/Optimizer.h"
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
};

// 12 sites //
/* std::vector<std::vector<int>> PROFILE::lattices = {{1,0},{3,1},{4,3},{0,4},{7,6},{9,7},{10,9},{6,10},// k_x
        {2,0},{6,2},{8,6},{0,8},{5,3},{9,5},{11,9},{3,11},// k_y
        {1,2},{11,1},{7,8},{5,7},{4,5},{8,4},{10,11},{2,10}}; //k_xy
*/

// 18 sites //
/* std::vector<std::vector<int>> PROFILE::lattices = {{1,0},{8,1},{9,8},{15,9},{17,15},{0,17},{4,3},{11,4},{6,11},{12,6},{14,12},{3,14},// k_x
        {2,3},{8,2},{7,8},{12,7},{13,12},{0,13},{5,0},{11,5},{10,11},{15,10},{16,15},{3,16},// k_y
        {2,1},{4,2},{5,4},{1,5},{7,6},{9,7},{10,9},{6,10},{14,13},{16,14},{17,16},{13,17}}; //k_xy
*/

// 24 sites //
std::vector<std::vector<int>> PROFILE::lattices = {{1,0},{6,1},{8,6},{15,8},{16,15},{21,16},{23,21},{0,23},{4,3},{9,4},{11,9},{12,11},{13,12},{18,13},{20,18},{3,20},// k_x
        {2,3},{6,2},{7,6},{12,7},{5,0},{9,5},{10,9},{15,10},{14,15},{18,14},{19,18},{0,19},{17,12},{21,17},{22,21},{3,22},// k_y
        {2,1},{4,2},{5,4},{1,5},{8,7},{10,8},{11,10},{7,11},{14,13},{16,14},{17,16},{13,17},{20,19},{22,20},{23,22},{19,23}}; // k_xy


// 30 sites //
/* std::vector<std::vector<int>> PROFILE::lattices = {{4,3},{9,4},{10,9},{1,0},{6,1},{7,6},{15,7},{16,15},{12,10},{13,12},{21,13},{22,21},{24,22},{18,16},{19,18},{27,19},{28,27},{0,28},{25,24},{3,25}, // k_x
        {5,3},{6,5},{8,6},{12,8},{14,12},{18,14},{20,18},{24,20},{26,24},{0,26},{2,0},{9,2},{11,9},{15,11},{17,15},{21,17},{23,21},{27,23},{29,27},{3,29}, // k_y
        {1,2},{5,1},{4,5},{2,4},{8,10},{7,8},{11,7},{10,11},{14,16},{13,14},{17,13},{16,17},{19,20},{23,19},{22,23},{20,22},{25,26},{29,25},{28,29},{26,28}}; // k_xy
*/

// std::vector<std::vector<int>> PROFILE::lattices = make_lattice(6);

using EIGEN = EigenSetRandomPot<LATTICE,PROFILE>;
using EQUATION = Gauss<EIGEN,PROFILE>;
using OPTIMIZER = Relaxation_fixedNum<EQUATION,PROFILE>;
using OPTIMIZER0 = MultiRF_fixedNum<EQUATION,PROFILE>;

double scheduler(const double U, const double X, double lambda);
std::string remove_zeros_in_str(const double val);

int main(int argc, char* argv[])
{
    std::vector<pair_t> options, defaults;
    // env; explanation of env
    options.push_back(pair_t("U", "interaction strength"));
    options.push_back(pair_t("mi", "initial guess of chemical potential"));
    options.push_back(pair_t("di", "initial guess of gap"));
    options.push_back(pair_t("Nsam", "# of disorder samples"));
    options.push_back(pair_t("X", "disorder strength"));
    options.push_back(pair_t("lambda", "hyper parameter to tune total Num"));
    options.push_back(pair_t("IsReInit", "turn on Re-initializing mi & di"));
    options.push_back(pair_t("path", "directory to load and save files"));
    // env; default value
    defaults.push_back(pair_t("Nsam", "1"));
    defaults.push_back(pair_t("X", "0"));
    defaults.push_back(pair_t("lambda", "0.1"));
    defaults.push_back(pair_t("IsReInit", "1"));
    defaults.push_back(pair_t("path", "."));
    // parser for arg list
    argsparse parser(argc, argv, options, defaults);

    int ranks, nprocs, flag;

	MPIERR(MPI_Init(&argc,&argv)); 
	MPIERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
	MPIERR(MPI_Comm_rank(MPI_COMM_WORLD, &ranks));

    const int L = PROFILE::L, 
        L2 = PROFILE::L2,
        N_tot = PROFILE::N_tot;
    const int Nsam = parser.find<int>("Nsam"),
        IsReInit = parser.find<int>("IsReInit");
    const auto UArr = parser.mfind<double>("U");
    const double X = parser.find<double>("X"),
        mi = parser.find<double>("mi"),
        di = parser.find<double>("di"),
        lambda = parser.find<double>("lambda");
    if(ranks==0) parser.print(std::cout);

	std::random_device rn;
	std::mt19937 generator(rn());
    std::uniform_real_distribution<double> dist(-1.0,1.0);

    double gamma[L];

    EIGEN eigen(L,gamma);
    EQUATION equation(eigen,L,63);
    OPTIMIZER optimizer(equation);
    OPTIMIZER0 optimizer0(equation);

    for (const auto &U : UArr){
        std::string filename, Ustr= parser.find<>("U"); //remove_zeros_in_str(U);        
        filename = parser.find<>("path") + "/KagomeProfile-L" + std::to_string(L) + "U" + Ustr + "X" + parser.find<>("X") + "N" + std::to_string(N_tot) + ".dat"; 
#ifdef __OUTFILE__
        std::ofstream outfile;
        MPIERR(MPI_Barrier(MPI_COMM_WORLD));
        if (!std::ifstream(filename).is_open())
        {
            outfile.open(filename);
            if (ranks == 0)
                outfile << "#       seq{x_i, d_i, m_i}        Ntot       mg      " << std::endl;
        }
        else
            outfile.open(filename,std::ios::app);
        outfile.precision(10);
        MPIERR(MPI_Barrier(MPI_COMM_WORLD));
#else
        if (ranks == 0)
            std::cout << "#       seq{x_i, d_i, m_i}        Ntot       mg      " << std::endl;
        std::cout.precision(10);
#endif
        double n,mg,mi_,di_;
        std::vector<double> m(L,mi), d(L,di), dilUsite(L,U);
        std::fill(gamma,gamma+L,1.0);
        if (IsReInit)
        {
            mg = mi-U/2.*N_tot/(double)L;
            optimizer0(m,d,dilUsite,n,mg,ranks,nprocs);
            mi_ = m[0]; di_ = d[0]; 
            if(ranks==0) std::cout << "\r#   ---   Re-initialization Finish   ---   #" << std::endl;
        }
        else
        {
            n = N_tot;
            mi_ = mi;
            di_ = di;
        }

        double lambda_ = scheduler(U,X,lambda);
        for(int i = 0; i < Nsam; i++){
            flag = false; 
            mg = mi_-U/2.*n/(double)L; // initial global chemical potential

            if(ranks==0){
                for(int j=0;j<L;++j) 
                    gamma[j] = X*dist(generator);
            }
            MPIERR(MPI_Bcast(&gamma[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));

            eigen.set_xconf(gamma);
            flag = optimizer(m,d,dilUsite,n,mg,ranks,nprocs,lambda_);

            if(flag)
            {
                if(ranks==0) std::cerr<<"# !!!  "<<i<<"-th iteration is roll-back !!!"<<std::endl;
                i--; 
                m.assign(L,mi_);
                d.assign(L,di_);
                continue;
            }
#ifdef __OUTFILE__		
            if(ranks == 0)
            {
                for(int j = 0; j < L; j++) outfile << "\t" << gamma[j] << "\t" << std::abs(d[j]) << "\t" << m[j] << "\t";
                outfile << n << "\t" << mg << std::endl;
            }
#else
            if(ranks == 0)
            {
                for(int j = 0; j < L; j++) std::cout << "\t" << gamma[j] << "\t" << std::abs(d[j]) << "\t" << m[j] << "\t";
                std::cout << n << "\t" << mg << std::endl;
            }
#endif
            m.assign(L,mi_);
            d.assign(L,di_);
        }
#ifdef __OUTFILE__	
        outfile.close();
#endif
    }
	MPIERR(MPI_Finalize());

	return 0;
}

double scheduler(const double U, const double X, double lambda)
{
    double U0 = 3.0, X0 = 0.5;
    if (X>0.0)
        return lambda*U/U0*X/X0; 
    else
        return lambda;
}

std::string remove_zeros_in_str(const double val)
{
    std::string tmp = std::to_string(val);
    int zero_pos = tmp.find_last_not_of('0'), point_pos = tmp.find_last_not_of('.');
    if (zero_pos-point_pos == 1){
        tmp.erase(zero_pos + 2, std::string::npos);
    } else {
        tmp.erase(zero_pos + 1, std::string::npos);
    }
    return tmp;
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


