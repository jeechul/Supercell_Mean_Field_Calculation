#ifndef EQUATION_H
#define EQUATION_H

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <functional>
#include <mpi.h>
#include "Eigen_system.h"

#define MPIERR(X) { if(X != MPI_SUCCESS) { std::cout<<"ERROR!"<<__LINE__<<std::endl; exit(1); } }

template <typename EIGEN>
class Equations
{
    EIGEN eigen;
    const int L,L2;
public:
    Equations(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    void Gap_Num_eq(const double k_x, const double k_y, double * gaps, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);

        for (int i = 0; i < L; ++i)
        {
            std::vector<dcomplex> u(L);
            std::vector<dcomplex> v(L);
            
            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);
            
            dcomplex temp1 = arrmul(u,u) - arrmul(v,v) + dcomplex(1.0,0);
            nums[i] += temp1.real()*c;
        
            dcomplex temp2 = arrmul(u,v)+arrmul(v,u);
            gaps[i] += temp2.real()*c; 		
        }
    }

    void Num_eq(const double k_x, const double k_y, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);

        for (int k = 0; k < L; k++)
        {
            std::vector<dcomplex> u(L);
            std::vector<dcomplex> v(L);
            
            std::copy(&BdG[L2*k],&BdG[L2*k+L],&u[0]);
            std::copy(&BdG[L2*(L+k)],&BdG[L2*(L+k)+L],&v[0]);
            dcomplex temp = arrmul(u,u) - arrmul(v,v) + dcomplex(1.0,0);
            nums[k] += temp.real()*c;
        };
    }

    void Gap_eq(const double k_x, const double k_y, double * gaps, double c, const std::vector<double> m, const std::vector<double> d)
    {	
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2];
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);
        
        for(int i = 0; i < L; i++){
            std::vector<dcomplex> u(L);
            std::vector<dcomplex> v(L);
            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);
            dcomplex temp = arrmul(u,v)+arrmul(v,u);
            gaps[i] += temp.real()*c; 		
        }
    }

    double W_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0)
    {
        const bool IsNotDev = true, IsTranspose = false;
        double result = 0;
        MatrixC BdG(L2); 
        double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'N',IsNotDev,IsTranspose);
        
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/dilute[i]+(m[i]-m0)*(m[i]-m0)/dilute[i];
        }

        return result;
    }

    double W_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double U, const double m0)
    {
        const bool IsNotDev = true, IsTranspose = false;
        double result = 0;
        MatrixC BdG(L2); 
        double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'N',IsNotDev,IsTranspose);
        
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/U+(m[i]-m0)*(m[i]-m0)/U;
        }

        return result;
    }

    std::pair<double,double> SFW_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double V, const double beta)
    {
        const bool IsTranspose = false;
        MatrixC g_T(L), g(L), BdG(L2), Psi(L2), G_T(L2);
        double e[L], E[L2];

        eigen.Single(k_x,k_y,e,g_T,'V',true,IsTranspose);
        eigen.Quasi(k_x,k_y,m,d,E,Psi,'V',true);

        g = g_T.Hermitian_transpose();
        G_T.put_Block(g_T,0,L,0,L);
        G_T.put_Block(g_T,L,L2,L,L2);
        BdG = G_T.zgemm(1.0,Psi);

        MatrixC dh(L);
        eigen.Make_dH0(k_x,k_y,dh);
        MatrixC g_dh_g = (g_T.zgemm(1.0,dh)).zgemm(1.0,g);

        MatrixC M_conv(L2), M_geom(L2);
        dcomplex SFW_conv=0, SFW_geom=0;
        /*
        for (int idx=0;idx<L2*L2;++idx){
            int i=idx/L2, j=idx%L2;
            dcomplex M_conv_p=0, M_conv_m=0;
            dcomplex M_geom_p=0, M_geom_m=0;
            
            for(int m=0;m<L;m++)
            for(int n=0;n<L;n++){
                if(m==n){ M_conv_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
                else{ M_geom_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
            }

            for(int q=L;q<L2;q++)
            for(int p=L;p<L2;p++){
                if(p==q){ M_conv_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
                else{ M_geom_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
            }

            //M_conv(i,j) = M_conv_p*M_conv_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
            //M_geom(i,j) = M_geom_p*M_geom_m;
            M_conv(i,j) = M_conv_p*M_conv_m; 
            M_geom(i,j) = M_geom_p*M_geom_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
        }
        */
        MatrixC g_dh_g_diag(L), g_dh_g_offdiag = g_dh_g;
        g_dh_g_diag.fill_diagonal(g_dh_g.diag());
        g_dh_g_offdiag.drop_all_diag();
        MatrixC BdG_p(L,L2), BdG_m(L,L2);
        BdG.pull_Block(BdG_p,0,L,0,L2);
        BdG.pull_Block(BdG_m,L,L2,0,L2);
        MatrixC BdG_p_T = BdG_p.Hermitian_transpose(), BdG_m_T = BdG_m.Hermitian_transpose();
        MatrixC M_conv_p_mat = (BdG_p_T.zgemm(1.0,g_dh_g_diag)).zgemm(1.0,BdG_p),
            M_geom_p_mat = (BdG_p_T.zgemm(1.0,g_dh_g_offdiag)).zgemm(1.0,BdG_p);
        MatrixC M_conv_m_mat = (BdG_m_T.zgemm(1.0,g_dh_g_diag)).zgemm(1.0,BdG_m),
            M_geom_m_mat = (BdG_m_T.zgemm(1.0,g_dh_g_offdiag)).zgemm(1.0,BdG_m);
        MatrixC M_conv_m_mat_t = M_conv_m_mat.transpose(), M_geom_m_mat_t = M_geom_m_mat.transpose();
        M_conv = M_conv_p_mat*M_conv_m_mat_t;
        M_geom = M_geom_p_mat*M_geom_m_mat_t + M_conv_p_mat*M_geom_m_mat_t + M_geom_p_mat*M_conv_m_mat_t;
        
    	for(int i=0;i<L;i++)
        {
            double coef1 = beta/(2.0*std::pow(std::cosh(beta*E[L2-1-i]/2.0),2))*-4.0,
                coef2 = std::tanh(beta*E[L2-1-i]/2.)/E[L2-1-i]*-2.0;
            SFW_conv += coef1*M_conv(i,i)+coef2*(M_conv(L2-1-i,i)+M_conv(i,L2-1-i));	
            SFW_geom += coef1*M_geom(i,i)+coef2*(M_geom(L2-1-i,i)+M_geom(i,L2-1-i));
            for(int j=i+1;j<L;j++)
            {
                double coef3 = (std::tanh(beta*E[L2-1-i]/2.)+std::tanh(beta*E[L2-1-j]/2.))/(E[L2-1-i]+E[L2-1-j])*-2.0,
                    coef4 = (std::tanh(beta*E[L+i]/2.)+std::tanh(beta*E[L+j]/2.))/(E[L+i]+E[L+j])*-2.0;
                SFW_conv += coef3*(M_conv(i,L2-1-j)+M_conv(L2-1-j,i))+coef4*(M_conv(L-1-i,L+j)+M_conv(L+j,L-1-i));
                SFW_geom += coef3*(M_geom(i,L2-1-j)+M_geom(L2-1-j,i))+coef4*(M_geom(L-1-i,L+j)+M_geom(L+j,L-1-i));
            }
        }

        return std::pair<double,double>(std::real(SFW_conv)/V,std::real(SFW_geom)/V);
    }

    std::pair<double,double> SFW_xy_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double V, const double beta)
    {
        const bool IsTranspose = false;
        MatrixC g_T(L), g(L), BdG(L2), Psi(L2), G_T(L2);
        double e[L], E[L2];

        eigen.Single(k_x,k_y,e,g_T,'V',true,IsTranspose);
        eigen.Quasi(k_x,k_y,m,d,E,Psi,'V',true);

        g = g_T.Hermitian_transpose();
        G_T.put_Block(g_T,0,L,0,L);
        G_T.put_Block(g_T,L,L2,L,L2);
        BdG = G_T.zgemm(1.0,Psi);

        MatrixC dhx(L), dhy(L);
        eigen.Make_dH0(k_x,k_y,dhx,"x");
        eigen.Make_dH0(k_x,k_y,dhy,"y");
        MatrixC g_dhx_g = (g_T.zgemm(1.0,dhx)).zgemm(1.0,g),
                g_dhy_g = (g_T.zgemm(1.0,dhy)).zgemm(1.0,g);

        MatrixC M_conv(L2), M_geom(L2);
        dcomplex SFW_conv=0, SFW_geom=0;

        for (int idx=0;idx<L2*L2;++idx){
            int i=idx/L2, j=idx%L2;
            dcomplex M_conv_p_x=0, M_conv_p_y=0, M_conv_m_x=0, M_conv_m_y=0;
            dcomplex M_geom_p_x=0, M_geom_p_y=0, M_geom_m_x=0, M_geom_m_y=0;

            for(int m=0;m<L;m++)
            for(int n=0;n<L;n++){
                if(m==n){
                    M_conv_p_x += std::conj(BdG(m,i))*BdG(n,j)*g_dhx_g(m,n);
                    M_conv_p_y += std::conj(BdG(m,i))*BdG(n,j)*g_dhy_g(m,n);
                }
                else{
                    M_geom_p_x += std::conj(BdG(m,i))*BdG(n,j)*g_dhx_g(m,n);
                    M_geom_p_y += std::conj(BdG(m,i))*BdG(n,j)*g_dhy_g(m,n);
                }
            }

            for(int q=L;q<L2;q++)
            for(int p=L;p<L2;p++){
                if(p==q){
                    M_conv_m_x += std::conj(BdG(q,j))*BdG(p,i)*g_dhx_g(q-L,p-L);
                    M_conv_m_y += std::conj(BdG(q,j))*BdG(p,i)*g_dhy_g(q-L,p-L);
                }
                else{
                    M_geom_m_x += std::conj(BdG(q,j))*BdG(p,i)*g_dhx_g(q-L,p-L);
                    M_geom_m_y += std::conj(BdG(q,j))*BdG(p,i)*g_dhy_g(q-L,p-L);
                }
            }
            //M_conv(i,j) = M_conv_p_x*M_conv_m_y + M_conv_p_y*M_conv_m_x;
            //M_geom(i,j) = (M_geom_p_x*M_geom_m_y+M_conv_p_x*M_geom_m_y+M_geom_p_x*M_conv_m_y)
            //            + (M_geom_p_y*M_geom_m_x+M_conv_p_y*M_geom_m_x+M_geom_p_y*M_conv_m_x);
            M_conv(i,j) = 2.0*M_conv_p_x*M_conv_m_y;
            M_geom(i,j) = 2.0*(M_geom_p_x*M_geom_m_y+M_conv_p_x*M_geom_m_y+M_geom_p_x*M_conv_m_y);
        }

        for(int i=0;i<L;i++)
        {
            double coef1 = beta/(2.0*std::pow(std::cosh(beta*E[L2-1-i]/2.0),2))*-2.0,
                coef2 = std::tanh(beta*E[L2-1-i]/2.)/E[L2-1-i]*-1.0;
            SFW_conv += coef1*M_conv(i,i)+coef2*(M_conv(L2-1-i,i)+M_conv(i,L2-1-i));
            SFW_geom += coef1*M_geom(i,i)+coef2*(M_geom(L2-1-i,i)+M_geom(i,L2-1-i));
            for(int j=i+1;j<L;j++)
            {
                double coef3 = (std::tanh(beta*E[L2-1-i]/2.)+std::tanh(beta*E[L2-1-j]/2.))/(E[L2-1-i]+E[L2-1-j])*-1.0,
                    coef4 = (std::tanh(beta*E[L+i]/2.)+std::tanh(beta*E[L+j]/2.))/(E[L+i]+E[L+j])*-1.0;
                SFW_conv += coef3*(M_conv(i,L2-1-j)+M_conv(L2-1-j,i))+coef4*(M_conv(L-1-i,L+j)+M_conv(L+j,L-1-i));
                SFW_geom += coef3*(M_geom(i,L2-1-j)+M_geom(L2-1-j,i))+coef4*(M_geom(L-1-i,L+j)+M_geom(L+j,L-1-i));
            }
        }

        return std::pair<double,double>(std::real(SFW_conv)/V,std::real(SFW_geom)/V);
    }
};


template <typename EIGEN>
class GreenFunction
{
    EIGEN eigen;
    const int L,L2;
public:
    GreenFunction(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    void single_green_function(const double k_x, const double k_y, double * ldos, const std::vector<double> Ws, const int site)
    {	
        const bool IsNotDev = true;
        MatrixC Psi(L); double E[L];
        eigen.Single(k_x,k_y,E,Psi,'V',IsNotDev);
       
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                double prob = std::norm(Psi[L*site+j]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],0.001)*M_PI));
            }
        }
    }

    void manybody_green_function(const double k_x, const double k_y, double * ldos, const std::vector<double> Ws, const int site, const std::vector<double> m, const std::vector<double> d)
    {	
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2];
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);
       
        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L2;++j)
            {
                double prob = std::norm(BdG[L2*site+j]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],0.005)*M_PI));
            }
        }
    }
};


template <typename EIGEN>
class SingleinReal
{
    EIGEN eigen;
    const int L;
public:
    SingleinReal(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size)
    {}

    void onsiteDOS(double * ldos, const std::vector<double> Ws, const int site)
    {
        const bool IsNotDev = true;
        double * PsiT = new double[L*L]; double E[L];
        eigen.Single0(E,PsiT,'V',IsNotDev);

        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                double prob = std::norm(PsiT[L*j+site]);
                ldos[i] -= std::imag(prob/(dcomplex(Ws[i]-E[j],0.001)*M_PI*(double)L));
            }
        }
        delete[] PsiT;
    }

    void totalDOS(double * ldos, const std::vector<double> Ws)
    {
        const bool IsNotDev = true;
        double * PsiT = new double[L*L]; double E[L];
        eigen.Single0(E,PsiT,'N',IsNotDev);

        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                ldos[i] -= std::imag(1.0/(dcomplex(Ws[i]-E[j],0.01)*M_PI*(double)L));
            }
        }
        delete[] PsiT;
    }

    void totalDOS_global_mem(double * PsiT, double * ldos, const std::vector<double> Ws)
    {
        const bool IsNotDev = true;
        double E[L];
        eigen.Single0(E,PsiT,'N',IsNotDev);

        for (int i=0;i<Ws.size();++i)
        {
            for (int j=0;j<L;++j)
            {
                ldos[i] -= std::imag(1.0/(dcomplex(Ws[i]-E[j],0.01)*M_PI*(double)L));
            }
        }
    }

    double occupation(double * ns, int Nt)
    {
        const bool IsNotDev = true;
        double * PsiT = new double[L*L]; double E[L];
        eigen.Single0(E,PsiT,'V',IsNotDev);

        std::vector<double> Er;
        std::map<double,std::vector<int>> Eid;

        for (int i=0;i<L;++i)
        {
            double e = std::round(E[i]*1e7)/1e7;
            if (std::find(Er.begin(),Er.end(),e) == Er.end()) {
                Er.push_back(e);
                Eid.insert({e,std::vector<int>(1,i)});
            }
            else {
                Eid[e].push_back(i);
            }
        }

        double sumE = 0.0, Nacc = 0.0;
        for (const auto e : Er)
        {
            int Nd = Eid[e].size();

            double fac = 1.0;
            if ((Nt-Nacc-Nd)<0) fac = (Nt-Nacc)/(double)Nd;

            for (int site=0; site<L; ++site)
            {
                for (const auto id : Eid[e])
                {
                    double prob = std::norm(PsiT[L*id+site])*fac;
                    ns[site] += prob;
                    Nacc += prob;
                }
            }
            sumE += e*Nd*fac;
        }

        delete[] PsiT;
        for (const auto e : Er){ 
            Eid[e].clear();
            std::vector<int>().swap(Eid[e]);
        }
        Eid.clear();

        return sumE;
    }

    double InverseParticipationNumber_global_mem(double * PsiT, std::vector<double> &ipns, std::vector<double> &Er)
    {
        const bool IsNotDev = true;
        double E[L];
        eigen.Single0(E,PsiT,'V',IsNotDev);

        std::map<double,std::vector<int>> Eid;

        for (int i=0;i<L;++i)
        {
            double e = std::round(E[i]*1e7)/1e7;
            if (std::find(Er.begin(),Er.end(),e) == Er.end()) {
                Er.push_back(e);
                Eid.insert({e,std::vector<int>(1,i)});
            }
            else {
                Eid[e].push_back(i);
            } 
        }

        double ipn_avg = 0;
        unsigned long long longL = L;
        for (const auto e : Er)
        {
            int Nd = Eid[e].size();
            double fac = 1/(double)Nd;

            double tmp_ipn = 0;
            for (const auto id : Eid[e])
            {
                for (int i=0;i<L;++i)
                {
                    tmp_ipn += PsiT[id*longL+i]*PsiT[id*longL+i]*PsiT[id*longL+i]*PsiT[id*longL+i];
                }
            }
            ipns.push_back(tmp_ipn*fac);
            ipn_avg += tmp_ipn*fac;
        }
        ipn_avg = ipn_avg/(double)(Er.size());

        for (const auto e : Er){ 
            Eid[e].clear();
            std::vector<int>().swap(Eid[e]);
        }
        Eid.clear();

        return ipn_avg;
    }

    void Prob_psi2_global_mem(double * PsiT, double * E, const std::vector<double> &Ws, std::vector<std::vector<double>> &Pr)
    {
        const bool IsNotDev = true;
        eigen.Single0(E,PsiT,'V',IsNotDev);

        unsigned long long longL = L;
        const double logl = 0.5*std::log(L);

        for (int id=0;id<longL;++id)
        {
            for (int j=0;j<longL;++j)
            {
                for (int i=0;i<Ws.size();++i){
                    double scaled_psi2 = -std::log(PsiT[id*longL+j]*PsiT[id*longL+j])/logl;
                    Pr[id][i] -= std::imag(1.0/(dcomplex(Ws[i]-scaled_psi2,5e-4)*M_PI*(double)L));
                }
            }
        }
    }
};


template <typename EIGEN, typename PROFILE>
class Gauss : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, xg, wg;
    const double kx_intv[2], ky_intv[2], RBZarea;
public:
    Gauss(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    kx_intv{-M_PI/2.*PROFILE::RBZ_x,M_PI/2.*PROFILE::RBZ_x},
    ky_intv{-M_PI/2.*PROFILE::RBZ_y,M_PI/2.*PROFILE::RBZ_y}
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        GaussLobattoPoints(Ngrid,xg,wg);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (kx_intv[0]+kx_intv[1])/2.+(kx_intv[1]-kx_intv[0])*xg[i]/2.;
            k_y[i] = (ky_intv[0]+ky_intv[1])/2.+(ky_intv[1]-ky_intv[0])*xg[i]/2.;
        }
    }

    inline int get_L() const { return L; }
    inline Gauss& operator=(const Gauss &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs) 
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }

    std::pair<double,double> SFW_xy (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

            double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            std::pair<double,double> sfw = this->SFW_xy_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class Trapz : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, wg;
    const double dk_x, dk_y, RBZarea;
public:
    Trapz(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        wg.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
                
            if (i > 0 && i < Ngrid-1)
                wg[i] = 2.0;
            else 
                wg[i] = 1.0;
        }
    }

    inline int get_L() const { return L; }
    inline Trapz& operator=(const Trapz &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],dk_x*dk_y*0.25*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class Simp : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, wg;
    const double dk_x, dk_y, RBZarea;
public:
    Simp(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        wg.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
                
            if (i > 0 && i < Ngrid-1)
            {
                if (i%2==1)
                    wg[i] = 4.0/3.0;
                else
                    wg[i] = 2.0/3.0;
            }
            else
            {
                wg[i] = 1.0/3.0;
            }
        }
    }

    inline int get_L() const { return L; }
    inline Simp& operator=(const Simp &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],dk_x*dk_y*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],dk_x*dk_y*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],dk_x*dk_y*wg[x]*wg[y],m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*wg[x]*wg[y];
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = dk_x*dk_y*0.25*wg[x]*wg[y];
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class AdaptiveSimp : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_y, wg;
    const double dk_y, RBZarea, k_x_min, k_x_max;

    void simpson_rule(std::function<MatrixD(double)> &func, const double a, const MatrixD &fa, const double b, const MatrixD &fb, double &m, MatrixD &fm, MatrixD &res)
    {
        m = (a+b)/2.0;
        fm = func(m);
        res = (b-a)/6.0*(fa+(4.0*fm)+fb);
    }

    MatrixD recursive_simpson_rule(std::function<MatrixD(double)> &func, const int Nf, const double a, const MatrixD &fa, const double b, const MatrixD &fb, const double tol, const MatrixD &whole, const double &m, const MatrixD &fm, int depth)
    {
        double lm, rm;
        MatrixD lfm(1,Nf), left(1,Nf), rfm(1,Nf), right(1,Nf);
        simpson_rule(func,a,fa,m,fm,lm,lfm,left);
        simpson_rule(func,m,fm,b,fb,rm,rfm,right);
        MatrixD delta = left+right-whole;
        if (depth <=0 || abs(*std::max_element(delta.data().begin(),delta.data().end())) <= 15.*tol) {
            return (left+right)+(delta/15.0);
        } else {
            recursive_simpson_rule(func,Nf,a,fa,m,fm,tol/2.,left,lm,lfm,depth-1) + recursive_simpson_rule(func,Nf,m,fm,b,fb,tol/2.,right,rm,rfm,depth-1);
        }
    }

    MatrixD stacking_simpson_rule(std::function<MatrixD(double)> &func, const int Nf, const double a, const MatrixD &fa, const double b, const MatrixD &fb, const double tol, const MatrixD &whole, const double &m, const MatrixD &fm, int depth)
    {
        std::vector<double> stack_a = {a}, stack_b = {b}, stack_m = {m}, stack_tol = {tol};
        std::vector<int> stack_depth = {depth};
        std::vector<MatrixD> stack_fa = {fa}, stack_fb = {fb}, stack_fm = {fm}, stack_whole = {whole};
        MatrixD result(1,Nf);

        while (!stack_a.empty()) {
            double a_ = stack_a.back(), b_ = stack_b.back(), m_ = stack_m.back(), tol_ = stack_tol.back();
            int depth_ = stack_depth.back();
            stack_a.pop_back(); stack_b.pop_back(); stack_m.pop_back(); stack_tol.pop_back(); stack_depth.pop_back();

            MatrixD fa_ = stack_fa.back(), fb_ = stack_fb.back(), fm_ = stack_fm.back(), whole_ = stack_whole.back();
            stack_fa.pop_back(); stack_fb.pop_back(); stack_fm.pop_back(); stack_whole.pop_back();

            double lm, rm;
            MatrixD lfm(1,Nf), left(1,Nf), rfm(1,Nf), right(1,Nf);
            simpson_rule(func,a_,fa_,m_,fm_,lm,lfm,left);
            simpson_rule(func,m_,fm_,b_,fb_,rm,rfm,right);

            MatrixD delta = left+right-whole_;
            if (abs(*std::max_element(delta.data().begin(),delta.data().end())) <= 15.*tol_ || depth_ <= 0) {
                result += (left+right)+(delta/15.0);
            } else {
                stack_a.push_back(a_); stack_b.push_back(m_); stack_m.push_back(lm); stack_tol.push_back(tol_/2.0); stack_depth.push_back(depth_-1);
                stack_fa.push_back(fa_); stack_fb.push_back(fm_); stack_fm.push_back(lfm); stack_whole.push_back(left);

                stack_a.push_back(m_); stack_b.push_back(b_); stack_m.push_back(rm); stack_tol.push_back(tol_/2.0); stack_depth.push_back(depth_-1);
                stack_fa.push_back(fm_); stack_fb.push_back(fb_); stack_fm.push_back(rfm); stack_whole.push_back(right);
            }
        }

        return result;
    }
public:
    AdaptiveSimp(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid), // Ngrid must be 2^n+1 (n is an integer)
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y),
    k_x_min(-Ngrid/(2.*(Ngrid-1))*M_PI*PROFILE::RBZ_x),
    k_x_max((Ngrid-2.)/(2.*(Ngrid-1))*M_PI*PROFILE::RBZ_x)
    {
        k_y.resize(Ngrid);
        wg.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;

            if (i > 0 && i < Ngrid-1)
            {
                if (i%2==1)
                    wg[i] = 4.0/3.0;
                else
                    wg[i] = 2.0/3.0;
            }
            else
            {
                wg[i] = 1.0/3.0;
            }
        }
    }

    inline int get_L() const { return L; }
    inline AdaptiveSimp& operator=(const AdaptiveSimp &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int y=ranks;y<Ngrid;y+=nprocs){
            std::function<MatrixD(double)> func = [this,y,&m,&d](double k_x){
                MatrixD gaps_nums(1,2*(this->L));
                this->Gap_Num_eq(k_x,(this->k_y)[y],&gaps_nums[0],&gaps_nums[L],1.0,m,d);
                return gaps_nums;
            };
            double m;
            MatrixD fmin = func(k_x_min), fmax = func(k_x_max), fm(1,2*L), whole(1,2*L);
            simpson_rule(func,k_x_min,fmin,k_x_max,fmax,m,fm,whole);
            MatrixD P_gaps_nums = stacking_simpson_rule(func,2*L,k_x_min,fmin,k_x_max,fmax,1e-8,whole,m,fm,12);

            for (int i=0;i<L;++i) {
                Pgaps[i] += P_gaps_nums[i]*wg[y]*dk_y;
                Pnums[i] += P_gaps_nums[L+i]*wg[y]*dk_y;
            }
        }
        MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
            }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int y=ranks;y<Ngrid;y+=nprocs){
            std::function<MatrixD(double)> func = [this,y,&m,&d](double k_x){
                MatrixD Temp_nums(1,this->L);
                this->Num_eq(k_x,(this->k_y)[y],Temp_nums[0],1.0,m,d);
                return Temp_nums;
            };
            double m;
            MatrixD fmin = func(k_x_min), fmax = func(k_x_max), fm(1,L), whole(1,L);
            simpson_rule(func,k_x_min,fmin,k_x_max,fmax,m,fm,whole);
            MatrixD Tnums = stacking_simpson_rule(func,L,k_x_min,fmin,k_x_max,fmax,1e-6,whole,m,fm,15);

            for (int i=0;i<L;++i) {
                Pnums[i] += Tnums[i]*wg[y]*dk_y;
            }
        }
        MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){
                nums[i] = nums[i]/RBZarea;
            }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0);

        for(int y=ranks;y<Ngrid;y+=nprocs){
            std::function<MatrixD(double)> func = [this,y,&m,&d](double k_x){
                MatrixD Temp_gaps(1,this->L);
                this->Gap_eq(k_x,(this->k_y)[y],Temp_gaps[0],1.0,m,d);
                return Temp_gaps;
            };
            double m;
            MatrixD fmin = func(k_x_min), fmax = func(k_x_max), fm(1,L), whole(1,L);
            simpson_rule(func,k_x_min,fmin,k_x_max,fmax,m,fm,whole);
            MatrixD Tgaps = stacking_simpson_rule(func,L,k_x_min,fmin,k_x_max,fmax,1e-6,whole,m,fm,15);

            for (int i=0;i<L;++i) {
                Pgaps[i] += Tgaps[i]*wg[y]*dk_y;
            }
        }
        MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){
                gaps[i] = gaps[i]/RBZarea;
            }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int y=ranks;y<Ngrid;y+=nprocs){
            std::function<MatrixD(double)> func = [this,y,&m,&d,&dilute,m0](double k_x){
                MatrixD Temp(1);
                Temp[0] = this->W_eq(k_x,(this->k_y)[y],m,d,dilute,m0);
                return Temp;
            };
            double m;
            MatrixD fmin = func(k_x_min), fmax = func(k_x_max), fm(1), whole(1);
            simpson_rule(func,k_x_min,fmin,k_x_max,fmax,m,fm,whole);
            MatrixD Tresult = stacking_simpson_rule(func,1,k_x_min,fmin,k_x_max,fmax,1e-8,whole,m,fm,12);

            Presult += Tresult[0]*wg[y]*dk_y;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int y=ranks;y<Ngrid;y+=nprocs){
            std::function<MatrixD(double)> func = [this,y,&m,&d,U,m0](double k_x){
                MatrixD Temp(1);
                Temp[0] = this->W_eq(k_x,(this->k_y)[y],m,d,U,m0);
                return Temp;
            };
            double m;
            MatrixD fmin = func(k_x_min), fmax = func(k_x_max), fm(1), whole(1);
            simpson_rule(func,k_x_min,fmin,k_x_max,fmax,m,fm,whole);
            MatrixD Tresult = stacking_simpson_rule(func,1,k_x_min,fmin,k_x_max,fmax,1e-6,whole,m,fm,15);

            Presult += Tresult[0]*wg[y]*dk_y;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int y=ranks;y<Ngrid;y+=nprocs){
            std::function<MatrixD(double)> func = [this,y,&m,&d,V,beta](double k_x){
                MatrixD Temp_sfw(1,2);
                std::pair<double,double> sfw_pair = this->SFW_eq(k_x,(this->k_y)[y],m,d,V,beta);
                Temp_sfw[0] = sfw_pair.first;
                Temp_sfw[1] = sfw_pair.second;
                return Temp_sfw;
            };
            double m;
            MatrixD fmin = func(k_x_min), fmax = func(k_x_max), fm(1,2), whole(1,2);
            simpson_rule(func,k_x_min,fmin,k_x_max,fmax,m,fm,whole);
            MatrixD Tresult = stacking_simpson_rule(func,2,k_x_min,fmin,k_x_max,fmax,1e-8,whole,m,fm,12);

            for (int i=0;i<2;++i) {
                Presult[i] += Tresult[i]*wg[y]*dk_y;
            }
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};


template <typename EIGEN, typename PROFILE>
class Stack : public GreenFunction<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y;
    const double dk_x, dk_y, RBZarea;
public:
    Stack(const EIGEN &eigen, const int size, const int Ngrid):
    GreenFunction<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
        }
    }

    inline int get_L() const { return L; }
    inline Stack& operator=(const Stack &Copy) { return *this; }

    void SingleLDOS (double * ldos, const int site, const std::vector<double> Ws, const int ranks, const int nprocs)
    {
        std::vector<double> Pldos(Ws.size(),0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->single_green_function(k_x[x],k_y[y],&Pldos[0],Ws,site);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pldos[0],&ldos[0],Ws.size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<Ws.size();i++){ 
                ldos[i] = ldos[i]/(Ngrid*Ngrid);
		    }
        }

        MPIERR(MPI_Bcast(&ldos[0],Ws.size(),MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void ManybodyLDOS (const std::vector<double> m, const std::vector<double> d, double * ldos, const int site, const std::vector<double> Ws, const int ranks, const int nprocs)
    {
        std::vector<double> Pldos(Ws.size(),0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->manybody_green_function(k_x[x],k_y[y],&Pldos[0],Ws,site,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pldos[0],&ldos[0],Ws.size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<Ws.size();i++){ 
                ldos[i] = ldos[i]/(Ngrid*Ngrid);
		    }
        }

        MPIERR(MPI_Bcast(&ldos[0],Ws.size(),MPI_DOUBLE,0,MPI_COMM_WORLD));
    }
};

template <typename EIGEN, typename PROFILE>
class ChernNumber
{
    EIGEN Eigen;
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y;
    const double dk_x, dk_y, RBZarea;
    std::vector<MatrixC> list_eigenvec;
    
    void set_list_eigenvec()
    {
        const bool IsNotDev = true, IsTranspose = false;

        for(int i=0;i<Ngrid*Ngrid;i++){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

            MatrixC Psi_T(L); double E[L];
            Eigen.Single(k_x[x],k_y[y],E,Psi_T,'V',IsNotDev,IsTranspose);

            list_eigenvec[i] = Psi_T;
        }
    }

    dcomplex berry_connection(const int i1, const int i2, const int n, const int m)
    {
        MatrixC Psi1_T(m-n,L), Psi2(L,m-n);
        list_eigenvec[i1].pull_Block(Psi1_T,n,m,0,L);
        (list_eigenvec[i2].Hermitian_transpose()).pull_Block(Psi2,0,L,n,m);

        MatrixC NonAbelBC = Psi1_T.zgemm(1.0,Psi2);

        return NonAbelBC.det(); 
    }
public:
    ChernNumber(const EIGEN &eigen, const int size, const int Ngrid):
    Eigen(eigen),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    dk_x(M_PI/((double)Ngrid)*PROFILE::RBZ_x),
    dk_y(M_PI/((double)Ngrid)*PROFILE::RBZ_y)
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        list_eigenvec.resize(Ngrid*Ngrid,MatrixC(L));

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_x;	// (something)*(a length of the range)
            k_y[i] = (2.*i - (Ngrid-1) - 1.)/(2.*(Ngrid-1)) * M_PI*PROFILE::RBZ_y;
        }
    }

    inline int get_L() const { return L; }
    inline ChernNumber& operator=(const ChernNumber &Copy) { return *this; }

    double FHS_method(const int n, const int m, const int ranks, const int nprocs)
    {
        set_list_eigenvec();
        MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        double Pchern = 0, chern = 0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid, x_ = (x+1)%Ngrid;
            int y = (int)(i/Ngrid), y_ = (y+1)%Ngrid;

            dcomplex bcArray1 = berry_connection(i, y*Ngrid+x_, n, m),
                bcArray2 = berry_connection(y*Ngrid+x_, y_*Ngrid+x_, n, m),
                bcArray3 = berry_connection(y_*Ngrid+x, y_*Ngrid+x_, n, m),
                bcArray4 = berry_connection(i, y_*Ngrid+x, n, m);

            dcomplex bc1 = bcArray1/std::abs(bcArray1),
                bc2 = bcArray2/std::abs(bcArray2),
                bc3 = bcArray3/std::abs(bcArray3),
                bc4 = bcArray4/std::abs(bcArray4);

            Pchern += std::log(bc1*bc2/(bc3*bc4)).imag();
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pchern,&chern,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            chern = chern/(M_PI*PROFILE::RBZ_x);
        }
        MPIERR(MPI_Bcast(&chern,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return chern;
    }
};


namespace FiniteT
{

template <typename EIGEN>
class Equations
{
    EIGEN eigen;
    const int L,L2;
    const double beta;
public:
    Equations(const EIGEN &Eigen, const int size, const double beta):
    eigen(Eigen),
    L(size), L2(size*2),
    beta(beta)
    {}

    void Gap_Num_eq(const double k_x, const double k_y, double * gaps, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);

        for (int i = 0; i < L; ++i)
        {
            std::vector<dcomplex> u(L), ubeta(L);
            std::vector<dcomplex> v(L), vbeta(L);
            
            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);

            for (int n=0; n<L; ++n) {
                double tanh_beta = tanh(beta*E[n]/2);
                ubeta[n] = u[n]*tanh_beta;
                vbeta[n] = v[n]*tanh_beta;
            }
            
            dcomplex temp1 = -arrmul(u,ubeta) + arrmul(v,vbeta) + dcomplex(1.0,0);
            nums[i] += temp1.real()*c;
        
            dcomplex temp2 = -arrmul(u,vbeta)-arrmul(v,ubeta);
            gaps[i] += temp2.real()*c; 		
        }
    }

    void Num_eq(const double k_x, const double k_y, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);

        for (int k = 0; k < L; k++)
        {
            std::vector<dcomplex> u(L), ubeta(L);
            std::vector<dcomplex> v(L), vbeta(L);
            
            std::copy(&BdG[L2*k],&BdG[L2*k+L],&u[0]);
            std::copy(&BdG[L2*(L+k)],&BdG[L2*(L+k)+L],&v[0]);

            for (int n=0; n<L; ++n) {
                double tanh_beta = tanh(beta*E[n]/2);
                ubeta[n] = u[n]*tanh_beta;
                vbeta[n] = v[n]*tanh_beta;
            }

            dcomplex temp = -arrmul(u,ubeta) + arrmul(v,vbeta) + dcomplex(1.0,0);
            nums[k] += temp.real()*c;
        };
    }

    void Gap_eq(const double k_x, const double k_y, double * gaps, double c, const std::vector<double> m, const std::vector<double> d)
    {	
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2];
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'V',IsNotDev);
        
        for(int i = 0; i < L; i++){
            std::vector<dcomplex> u(L), ubeta(L);
            std::vector<dcomplex> v(L), vbeta(L);

            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);

            for (int n=0; n<L; ++n) {
                double tanh_beta = tanh(beta*E[n]/2);
                ubeta[n] = u[n]*tanh_beta;
                vbeta[n] = v[n]*tanh_beta;
            }

            dcomplex temp = -arrmul(u,vbeta)-arrmul(v,ubeta);
            gaps[i] += temp.real()*c; 		
        }
    }

    double W_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0)
    {
        const bool IsNotDev = true, IsTranspose = false;
        double result = 0;
        MatrixC BdG(L2); 
        double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'N',IsNotDev,IsTranspose);
        
        for(int i = 0; i < L; i++){
            double cosh_beta = cosh(beta*E[i]/2); 
            result += -log(4*cosh_beta*cosh_beta)/beta-m[i]+d[i]*d[i]/dilute[i]+(m[i]-m0)*(m[i]-m0)/dilute[i];
        }

        return result;
    }

    double W_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double U, const double m0)
    {
        const bool IsNotDev = true, IsTranspose = false;
        double result = 0;
        MatrixC BdG(L2); 
        double E[L2]; 
        eigen.Quasi(k_x,k_y,m,d,E,BdG,'N',IsNotDev,IsTranspose);
        
        for(int i = 0; i < L; i++){ 
            double cosh_beta = cosh(beta*E[i]/2); 
            result += -log(4*cosh_beta*cosh_beta)/beta-m[i]+d[i]*d[i]/U+(m[i]-m0)*(m[i]-m0)/U;
        }

        return result;
    }

    std::pair<double,double> SFW_eq(const double k_x, const double k_y, const std::vector<double> m, const std::vector<double> d, const double V)
    {
        const bool IsTranspose = false;
        MatrixC g_T(L), g(L), BdG(L2), Psi(L2), G_T(L2);
        double e[L], E[L2];

        eigen.Single(k_x,k_y,e,g_T,'V',true,IsTranspose);
        eigen.Quasi(k_x,k_y,m,d,E,Psi,'V',true);

        g = g_T.Hermitian_transpose();
        G_T.put_Block(g_T,0,L,0,L);
        G_T.put_Block(g_T,L,L2,L,L2);
        BdG = G_T.zgemm(1.0,Psi);

        MatrixC dh(L);
        eigen.Make_dH0(k_x,k_y,dh);
        MatrixC g_dh_g = (g_T.zgemm(1.0,dh)).zgemm(1.0,g);

        MatrixC M_conv(L2), M_geom(L2);
        dcomplex SFW_conv=0, SFW_geom=0;

        for (int idx=0;idx<L2*L2;++idx){
            int i=idx/L2, j=idx%L2;
            dcomplex M_conv_p=0, M_conv_m=0;
            dcomplex M_geom_p=0, M_geom_m=0;
            
            for(int m=0;m<L;m++)
            for(int n=0;n<L;n++){
                if(m==n){ M_conv_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
                else{ M_geom_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
            }

            for(int q=L;q<L2;q++)
            for(int p=L;p<L2;p++){
                if(p==q){ M_conv_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
                else{ M_geom_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
            }

            //M_conv(i,j) = M_conv_p*M_conv_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
            //M_geom(i,j) = M_geom_p*M_geom_m; 
            M_conv(i,j) = M_conv_p*M_conv_m; 
            M_geom(i,j) = M_geom_p*M_geom_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
        }

    	for(int i=0;i<L;i++)
        {
            double coef1 = beta/(2.0*std::pow(std::cosh(beta*E[L2-1-i]/2.0),2))*-4.0,
                coef2 = std::tanh(beta*E[L2-1-i]/2.)/E[L2-1-i]*-2.0;
            SFW_conv += coef1*M_conv(i,i)+coef2*(M_conv(L2-1-i,i)+M_conv(i,L2-1-i));	
            SFW_geom += coef1*M_geom(i,i)+coef2*(M_geom(L2-1-i,i)+M_geom(i,L2-1-i));

            for(int j=i+1;j<L;j++)
            {
                double coef3 = (std::tanh(beta*E[L2-1-i]/2.)+std::tanh(beta*E[L2-1-j]/2.))/(E[L2-1-i]+E[L2-1-j])*-2.0,
                    coef4 = (std::tanh(beta*E[L+i]/2.)+std::tanh(beta*E[L+j]/2.))/(E[L+i]+E[L+j])*-2.0;
                SFW_conv += coef3*(M_conv(i,L2-1-j)+M_conv(L2-1-j,i))+coef4*(M_conv(L-1-i,L+j)+M_conv(L+j,L-1-i));
                SFW_geom += coef3*(M_geom(i,L2-1-j)+M_geom(L2-1-j,i))+coef4*(M_geom(L-1-i,L+j)+M_geom(L+j,L-1-i));

                double coef5 = (1/(1+exp(-beta*E[L2-1-j]))-1/(1+exp(-beta*E[L2-1-i])))/(-E[L2-1-i]+E[L2-1-j])*-4.0,
                    coef6 = (1/(1+exp(beta*E[L+j]))-1/(1+exp(beta*E[L+i])))/(E[L+i]-E[L+j])*-4.0;
                SFW_conv += coef5*(M_conv(i,j)+M_conv(j,i)) + coef6*(M_conv(L+i,L+j)+M_conv(L+j,L+i));
                SFW_geom += coef5*(M_geom(i,j)+M_geom(j,i)) + coef6*(M_geom(L+i,L+j)+M_geom(L+j,L+i));
            } 
        }

        return std::pair<double,double>(std::real(SFW_conv)/V,std::real(SFW_geom)/V);
    }
};


template <typename EIGEN, typename PROFILE>
class Gauss : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k_x, k_y, xg, wg;
    const double kx_intv[2], ky_intv[2], RBZarea, beta;
public:
    Gauss(const EIGEN &eigen, const int size, const double beta, const int Ngrid):
    Equations<EIGEN>(eigen, size, beta),
    Ngrid(Ngrid),
    L(size), L2(size*2),
    beta(beta),
    RBZarea(M_PI*M_PI*PROFILE::RBZ_x*PROFILE::RBZ_y),
    kx_intv{-M_PI/2.*PROFILE::RBZ_x,M_PI/2.*PROFILE::RBZ_x},
    ky_intv{-M_PI/2.*PROFILE::RBZ_y,M_PI/2.*PROFILE::RBZ_y}
    {
        k_x.resize(Ngrid);
        k_y.resize(Ngrid);
        GaussLobattoPoints(Ngrid,xg,wg);

        for(int i=0;i<Ngrid;i++){
            k_x[i] = (kx_intv[0]+kx_intv[1])/2.+(kx_intv[1]-kx_intv[0])*xg[i]/2.;
            k_y[i] = (ky_intv[0]+ky_intv[1])/2.+(ky_intv[1]-ky_intv[0])*xg[i]/2.;
        }
    }

    inline int get_L() const { return L; }
    inline Gauss& operator=(const Gauss &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_Num_eq(k_x[x],k_y[y],&Pgaps[0],&Pnums[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Num_eq(k_x[x],k_y[y],&Pnums[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs) 
    {
        std::vector<double> Pgaps(L,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);
            this->Gap_eq(k_x[x],k_y[y],&Pgaps[0],(ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k_x[x],k_y[y],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k_x[x],k_y[y],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int i=ranks;i<Ngrid*Ngrid;i+=nprocs){
            int x = i%Ngrid;
            int y = (int)(i/Ngrid);

		    double weight = (ky_intv[1]-ky_intv[0])*wg[y]/2.*(kx_intv[1]-kx_intv[0])*wg[x]/2.;
            std::pair<double,double> sfw = this->SFW_eq(k_x[x],k_y[y],m,d,V);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};

} // namespace FiniteT

namespace OneDimension
{

template <typename EIGEN>
class Equations
{
    EIGEN eigen;
    const int L,L2;
public:
    Equations(const EIGEN &Eigen, const int size):
    eigen(Eigen),
    L(size), L2(size*2)
    {}

    void Gap_Num_eq(const double k, double * gaps, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2]; 
        eigen.Quasi(k,m,d,E,BdG,'V',IsNotDev);

        for (int i = 0; i < L; ++i)
        {
            std::vector<dcomplex> u(L);
            std::vector<dcomplex> v(L);
            
            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);
            
            dcomplex temp1 = arrmul(u,u) - arrmul(v,v) + dcomplex(1.0,0);
            nums[i] += temp1.real()*c;
        
            dcomplex temp2 = arrmul(u,v)+arrmul(v,u);
            gaps[i] += temp2.real()*c; 		
        }
    }

    void Num_eq(const double k, double * nums, double c, const std::vector<double> m, const std::vector<double> d)
    {
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2]; 
        eigen.Quasi(k,m,d,E,BdG,'V',IsNotDev);

        for (int i = 0; i < L; i++)
        {
            std::vector<dcomplex> u(L);
            std::vector<dcomplex> v(L);
            
            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);
            dcomplex temp = arrmul(u,u) - arrmul(v,v) + dcomplex(1.0,0);
            nums[i] += temp.real()*c;
        };
    }

    void Gap_eq(const double k, double * gaps, double c, const std::vector<double> m, const std::vector<double> d)
    {	
        const bool IsNotDev = true;
        MatrixC BdG(L2); double E[L2];
        eigen.Quasi(k,m,d,E,BdG,'V',IsNotDev);
        
        for(int i = 0; i < L; i++){
            std::vector<dcomplex> u(L);
            std::vector<dcomplex> v(L);
            std::copy(&BdG[L2*i],&BdG[L2*i+L],&u[0]);
            std::copy(&BdG[L2*(L+i)],&BdG[L2*(L+i)+L],&v[0]);
            dcomplex temp = arrmul(u,v)+arrmul(v,u);
            gaps[i] += temp.real()*c; 		
        }
    }

    double W_eq(const double k, const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0)
    {
        const bool IsNotDev = true, IsTranspose = false;
        double result = 0;
        MatrixC BdG(L2); 
        double E[L2]; 
        eigen.Quasi(k,m,d,E,BdG,'N',IsNotDev,IsTranspose);
        
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/dilute[i]+(m[i]-m0)*(m[i]-m0)/dilute[i];
        }

        return result;
    }

    double W_eq(const double k, const std::vector<double> m, const std::vector<double> d, const double U, const double m0)
    {
        const bool IsNotDev = true, IsTranspose = false;
        double result = 0;
        MatrixC BdG(L2); 
        double E[L2]; 
        eigen.Quasi(k,m,d,E,BdG,'N',IsNotDev,IsTranspose);
        
        for(int i = 0; i < L; i++){ 
            result += E[i]-m[i]+d[i]*d[i]/U+(m[i]-m0)*(m[i]-m0)/U;
        }

        return result;
    }

    std::pair<double,double> SFW_eq(const double k, const std::vector<double> m, const std::vector<double> d, const double V, const double beta)
    {
        const bool IsTranspose = false;
        MatrixC g_T(L), g(L), BdG(L2), Psi(L2), G_T(L2);
        double e[L], E[L2];

        eigen.Single(k,e,g_T,'V',true,IsTranspose);
        eigen.Quasi(k,m,d,E,Psi,'V',true);

        g = g_T.Hermitian_transpose();
        G_T.put_Block(g_T,0,L,0,L);
        G_T.put_Block(g_T,L,L2,L,L2);
        BdG = G_T.zgemm(1.0,Psi);

        MatrixC dh(L);
        eigen.Make_dH0(k,dh);
        MatrixC g_dh_g = (g_T.zgemm(1.0,dh)).zgemm(1.0,g);

        MatrixC M_conv(L2), M_geom(L2);
        dcomplex SFW_conv=0, SFW_geom=0;

        for (int idx=0;idx<L2*L2;++idx){
            int i=idx/L2, j=idx%L2;
            dcomplex M_conv_p=0, M_conv_m=0;
            dcomplex M_geom_p=0, M_geom_m=0;
            
            for(int m=0;m<L;m++)
            for(int n=0;n<L;n++){
                if(m==n){ M_conv_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
                else{ M_geom_p += std::conj(BdG(m,i))*BdG(n,j)*g_dh_g(m,n); }
            }

            for(int q=L;q<L2;q++)
            for(int p=L;p<L2;p++){
                if(p==q){ M_conv_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
                else{ M_geom_m += std::conj(BdG(q,j))*BdG(p,i)*g_dh_g(q-L,p-L); }
            }

            //M_conv(i,j) = M_conv_p*M_conv_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
            //M_geom(i,j) = M_geom_p*M_geom_m; 
            M_conv(i,j) = M_conv_p*M_conv_m; 
            M_geom(i,j) = M_geom_p*M_geom_m+M_conv_p*M_geom_m+M_geom_p*M_conv_m; 
        }

    	for(int i=0;i<L;i++)
        {
            double coef1 = beta/(2.0*std::pow(std::cosh(beta*E[L2-1-i]/2.0),2))*-4.0,
                coef2 = std::tanh(beta*E[L2-1-i]/2.)/E[L2-1-i]*-2.0;
            SFW_conv += coef1*M_conv(i,i)+coef2*(M_conv(L2-1-i,i)+M_conv(i,L2-1-i));	
            SFW_geom += coef1*M_geom(i,i)+coef2*(M_geom(L2-1-i,i)+M_geom(i,L2-1-i));
            for(int j=i+1;j<L;j++)
            {
                double coef3 = (std::tanh(beta*E[L2-1-i]/2.)+std::tanh(beta*E[L2-1-j]/2.))/(E[L2-1-i]+E[L2-1-j])*-2.0,
                    coef4 = (std::tanh(beta*E[L+i]/2.)+std::tanh(beta*E[L+j]/2.))/(E[L+i]+E[L+j])*-2.0;
                SFW_conv += coef3*(M_conv(i,L2-1-j)+M_conv(L2-1-j,i))+coef4*(M_conv(L-1-i,L+j)+M_conv(L+j,L-1-i));
                SFW_geom += coef3*(M_geom(i,L2-1-j)+M_geom(L2-1-j,i))+coef4*(M_geom(L-1-i,L+j)+M_geom(L+j,L-1-i));
            }
        }

        return std::pair<double,double>(std::real(SFW_conv)/V,std::real(SFW_geom)/V);
    }
};

template <typename EIGEN, typename PROFILE>
class Gauss : public Equations<EIGEN>
{
    const int Ngrid, L, L2;
    std::vector<double> k, xg, wg;
    const double k_intv[2], RBZarea;
public:
    Gauss(const EIGEN &eigen, const int size, const int Ngrid):
    Equations<EIGEN>(eigen, size),
    Ngrid(Ngrid),
    k_intv{-M_PI/2.*PROFILE::RBZ,M_PI/2.*PROFILE::RBZ},
    L(size), L2(size*2),
    RBZarea(M_PI*PROFILE::RBZ)
    {
        k.resize(Ngrid);
        GaussLobattoPoints(Ngrid,xg,wg);

        for(int i=0;i<Ngrid;i++){
            k[i] = (k_intv[0]+k_intv[1])/2.+(k_intv[1]-k_intv[0])*xg[i]/2.;
        }
    }

    inline int get_L() const { return L; }
    inline Gauss& operator=(const Gauss &Copy) { return *this; }

    void Gap_Num (const std::vector<double> m, const std::vector<double> d, double * gaps, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pgaps(L,0), Pnums(L,0);

        for(int x=ranks;x<Ngrid;x+=nprocs){
            this->Gap_Num_eq(k[x],&Pgaps[0],&Pnums[0],(k_intv[1]-k_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Num (const std::vector<double> m, const std::vector<double> d, double * nums, const int ranks, const int nprocs)
    {
        std::vector<double> Pnums(L,0);

        for(int x=ranks;x<Ngrid;x+=nprocs){
            this->Num_eq(k[x],&Pnums[0],(k_intv[1]-k_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pnums[0],&nums[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                nums[i] = nums[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&nums[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    void Gap (const std::vector<double> m, const std::vector<double> d, double * gaps, const int ranks, const int nprocs) 
    {
        std::vector<double> Pgaps(L,0);

        for(int x=ranks;x<Ngrid;x+=nprocs){
            this->Gap_eq(k[x],&Pgaps[0],(k_intv[1]-k_intv[0])*wg[x]/2.,m,d);
        }
	    MPIERR(MPI_Barrier(MPI_COMM_WORLD));

        MPIERR(MPI_Reduce(&Pgaps[0],&gaps[0],L,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            for(int i=0;i<L;i++){ 
                gaps[i] = gaps[i]/RBZarea;
		    }
        }

        MPIERR(MPI_Bcast(&gaps[0],L,MPI_DOUBLE,0,MPI_COMM_WORLD));
    }

    double W (const std::vector<double> m, const std::vector<double> d, const std::vector<double> &dilute, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int x=ranks;x<Ngrid;x+=nprocs){
		    double weight = (k_intv[1]-k_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k[x],m,d,dilute,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    double W (const std::vector<double> m, const std::vector<double> d, const double U, const double m0, const int ranks, const int nprocs)
    {
        double Presult=0,result=0;

        for(int x=ranks;x<Ngrid;x+=nprocs){
		    double weight = (k_intv[1]-k_intv[0])*wg[x]/2.;
            Presult += this->W_eq(k[x],m,d,U,m0)*weight;
        }

        MPIERR(MPI_Reduce(&Presult,&result,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result = result/RBZarea;
        }

        MPIERR(MPI_Bcast(&result,1,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return result;
    }

    std::pair<double,double> SFW (const std::vector<double> m, const std::vector<double> d, const double V, const double beta, const int ranks, const int nprocs)
    {
        std::vector<double> Presult(2,0), result(2,0);

        for(int x=ranks;x<Ngrid;x+=nprocs){
		    double weight = (k_intv[1]-k_intv[0])*wg[x]/2.;
            std::pair<double,double> sfw = this->SFW_eq(k[x],m,d,V,beta);
            Presult[0] += sfw.first*weight;
            Presult[1] += sfw.second*weight;
        }

        MPIERR(MPI_Reduce(&Presult[0],&result[0],2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

        if(ranks==0){
            result[0] = result[0]/RBZarea;
            result[1] = result[1]/RBZarea;
        }

        MPIERR(MPI_Bcast(&result[0],2,MPI_DOUBLE,0,MPI_COMM_WORLD));

        return std::pair<double,double>(result[0],result[1]);
    }
};

} // namespace OneDimension

#endif
