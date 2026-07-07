#ifndef EIGEN_SYSTEM_H
#define EIGEN_SYSTEM_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include "Matrix_operator.h"

// Based on colum major
extern "C" {
void zheev_(const char* JOBZ, const char* UPLO, const int* N, dcomplex* A, const int* LDA, double* W, dcomplex* WORK, const int* LWORK, double* RWORK, int* INFO);
void dsyev_(const char* JOBZ, const char* UPLO, const int* N, double* A, const int* LDA, double* W, double* WORK, const int* LWORK, int* INFO);
}

MatrixC zheev(MatrixC H0, double * E, const int N, const char JOBZ, const bool IsTranspose)
{
	const char UPLO = 'L'; // if JOBZ is 'V', val and vec. if JOBZ is 'N', val only.
	const int LDA = N;
	std::vector<double> RWORK(3*N-2);
	std::vector<dcomplex> WORK(1);
	int LWORK = -1, INFO = 0;

	zheev_(&JOBZ, &UPLO, &N, &H0[0], &LDA, &E[0], &WORK[0], &LWORK, &RWORK[0], &INFO);
	LWORK = WORK[0].real();
	std::vector<dcomplex>(LWORK, dcomplex(0, 0)).swap(WORK);

	zheev_(&JOBZ, &UPLO, &N, &H0[0], &LDA, &E[0], &WORK[0], &LWORK, &RWORK[0], &INFO);

	if(IsTranspose){ return H0.Hermitian_transpose(); }
	else{ return H0; } 
};

void dsyev(double * H0, double * E, const int N, const char JOBZ)
{
	const char UPLO = 'L'; // if JOBZ is 'V', val and vec. if JOBZ is 'N', val only.
	const int LDA = N;
	std::vector<double> WORK(1);
	int LWORK = -1, INFO = 0;

	dsyev_(&JOBZ, &UPLO, &N, &H0[0], &LDA, &E[0], &WORK[0], &LWORK, &INFO);
	LWORK = WORK[0];
	std::vector<double>(LWORK, 0.0).swap(WORK);

	dsyev_(&JOBZ, &UPLO, &N, &H0[0], &LDA, &E[0], &WORK[0], &LWORK, &INFO);
};

template <typename LATTICE, typename PROFILE>
class EigenSet
{
    const int L, L2;
public:
    EigenSet(const int size):
    L(size), L2(size*2)
    {}

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,false);
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);
		Evec = zheev(h,Eval,L,JOBZ,Switch);
	}

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
		dsyev(Evec,Eval,L,JOBZ);
	}

	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);
		MatrixC H(L2);

		for (int i=0;i<L;++i) h(i,i) = -m[i];

		H.put_Block(h,0,L,0,L);
		H.put_Block(h.minus(),L,L2,L,L2);
		H.fill_diagonal(d,0,L);
		H.fill_diagonal(d,L,0); 

		eig_vec = zheev(H,eig_val,L2,JOBZ,Switch);
	}
};


template <typename LATTICE, typename PROFILE>
class EigenSetMCM // g^{up triangle} = g^{down triangle} : with inversion symmetry
{
    double *gamma;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetMCM(const int size, double *g, bool IsRev_=false):
    gamma(g),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *gamma_){ gamma = gamma_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0, const std::string dir = "x")
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,gamma,false,dir);
        if (IsRev) dH0.minus();
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);
        for(int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i];
        if (IsRev)
		    Evec = zheev(h.minus(),Eval,L,JOBZ,Switch);
        else
		    Evec = zheev(h,Eval,L,JOBZ,Switch);
	}

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec[i*L+i] = -2.0*gamma[i]*gamma[i];
		dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
	}

	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);
		MatrixC H(L2);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]+m[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -2.0*gamma[i]*gamma[i]-m[i];
        }

		H.put_Block(h,0,L,0,L);
		H.put_Block(h.minus(),L,L2,L,L2);
		H.fill_diagonal(d,0,L);
		H.fill_diagonal(d,L,0); 

		eig_vec = zheev(H,eig_val,L2,JOBZ,Switch);
	}
};


template <typename LATTICE, typename PROFILE>
class EigenSetRandomPot
{
    double *X;
    const int L, L2;
public:
    EigenSetRandomPot(const int size, double *X):
    X(X),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *X_){ X = X_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,false);
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);
        for (int i=0; i<L; ++i) h(i,i) = -X[i];
		Evec = zheev(h,Eval,L,JOBZ,Switch);
	}

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,IsNotDev);
        for (int i=0; i<L; ++i) Evec[i*L+i] = -X[i];
		dsyev(Evec,Eval,L,JOBZ);
	}

	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,IsNotDev),L);
        for (int i=0; i<L; ++i) h(i,i) = -X[i]-m[i];
		MatrixC H(L2);

		H.put_Block(h,0,L,0,L);
		H.put_Block(h.minus(),L,L2,L,L2);
		H.fill_diagonal(d,0,L);
		H.fill_diagonal(d,L,0); 

		eig_vec = zheev(H,eig_val,L2,JOBZ,Switch);
	}
};


template <typename LATTICE, typename PROFILE>
class EigenSetMCMonlyHop // g^{up triangle} = g^{down triangle} : with inversion symmetry
{
    double *gamma;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetMCMonlyHop(const int size, double *g, bool IsRev_=false):
    gamma(g),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *gamma_){ gamma = gamma_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat(PROFILE::lattices,gamma,false);
        if (IsRev) dH0.minus();
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);
        if (IsRev)
		    Evec = zheev(h.minus(),Eval,L,JOBZ,Switch);
        else
		    Evec = zheev(h,Eval,L,JOBZ,Switch);
	}

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_real(Evec,PROFILE::lattices,gamma,IsNotDev);
		dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
	}

	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat(PROFILE::lattices,gamma,IsNotDev),L);
		MatrixC H(L2);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = m[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -m[i];
        }

		H.put_Block(h,0,L,0,L);
		H.put_Block(h.minus(),L,L2,L,L2);
		H.fill_diagonal(d,0,L);
		H.fill_diagonal(d,L,0); 

		eig_vec = zheev(H,eig_val,L2,JOBZ,Switch);
	}
};


template <typename LATTICE, typename PROFILE>
class EigenSetBDM // g^{up triangle} = 1/g^{down triangle} : without inversion symmetry
{
    double *gamma, *X;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetBDM(const int size, double *g, double *X, bool IsRev_=false):
    gamma(g),
    X(X),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *X_){ X = X_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat_BDM(PROFILE::lattices,gamma,false);
        if (IsRev) dH0.minus();
    }

    void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
    {
        LATTICE geometry(k_x,k_y,L);
        MatrixC h(geometry.make_mat_BDM(PROFILE::lattices,gamma,IsNotDev),L);
        for(int i=0;i<L;++i){
            h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-X[i];
        }
        if (IsRev)
            Evec = zheev(h.minus(),Eval,L,JOBZ,Switch);
        else
            Evec = zheev(h,Eval,L,JOBZ,Switch);
    }

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
    {
        LATTICE geometry(0.0,0.0,L);
        geometry.make_mat_BDM_real(Evec,PROFILE::lattices,gamma,IsNotDev);
        for(int i=0;i<L;++i) Evec[i*L+i] = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-X[i];
        dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
    }

    void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
    {
        LATTICE geometry(k_x,k_y,L);
        MatrixC h(geometry.make_mat_BDM(PROFILE::lattices,gamma,IsNotDev),L);
        MatrixC H(L2);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])+m[i]+X[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -gamma[i]*gamma[i]-1.0/(gamma[i]*gamma[i])-m[i]-X[i];
        }

        H.put_Block(h,0,L,0,L);
        H.put_Block(h.minus(),L,L2,L,L2);
        H.fill_diagonal(d,0,L);
        H.fill_diagonal(d,L,0);

        eig_vec = zheev(H,eig_val,L2,JOBZ,Switch);
    }
};

template <typename LATTICE, typename PROFILE>
class EigenSetCosineHop // PRB 111, L020506 (2025)
{
    double *gamma, W;
    const int L, L2;
    const bool IsRev;
public:
    EigenSetCosineHop(const int size, double *g, double W, bool IsRev_=false):
    gamma(g),
    W(W),
    IsRev(IsRev_),
    L(size), L2(size*2)
    {}

    inline void set_xconf(double *gamma_){ gamma = gamma_; }

    void Make_dH0(const double k_x, const double k_y, MatrixC &dH0)
    {
        LATTICE geometry(k_x,k_y,L);
        dH0 = geometry.make_mat_cosine(PROFILE::lattices,gamma,W,false);
        if (IsRev) dH0.minus();
    }

	void Single(const double k_x, const double k_y, double * Eval, MatrixC &Evec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat_cosine(PROFILE::lattices,gamma,W,IsNotDev),L);
        if (IsRev)
		    Evec = zheev(h.minus(),Eval,L,JOBZ,Switch);
        else
		    Evec = zheev(h,Eval,L,JOBZ,Switch);
	}

    void Single0(double * Eval, double * Evec, const char JOBZ, bool IsNotDev)
	{
        LATTICE geometry(0.0,0.0,L);
		geometry.make_mat_cosine_real(Evec,PROFILE::lattices,gamma,W,IsNotDev);
		dsyev(Evec,Eval,L,JOBZ); // it is transposed and do not support reversed band
	}

	void Quasi(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, double * eig_val, MatrixC &eig_vec, const char JOBZ, bool IsNotDev, bool Switch=true)
	{
        LATTICE geometry(k_x,k_y,L);
		MatrixC h(geometry.make_mat_cosine(PROFILE::lattices,gamma,W,IsNotDev),L);
		MatrixC H(L2);

        if (IsRev){
            for(int i=0;i<L;++i) h(i,i) = m[i];
            h.minus();
        }
        else {
            for(int i=0;i<L;++i) h(i,i) = -m[i];
        }

		H.put_Block(h,0,L,0,L);
		H.put_Block(h.minus(),L,L2,L,L2);
		H.fill_diagonal(d,0,L);
		H.fill_diagonal(d,L,0); 

		eig_vec = zheev(H,eig_val,L2,JOBZ,Switch);
	}
};

#endif
