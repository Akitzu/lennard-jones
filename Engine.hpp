#include "Engine.h"
#include "ConfigFile.h"
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <omp.h>
#include <valarray>

// valarrays operators
template <typename T>
std::valarray<std::valarray<T>> operator*(double const &a1, std::valarray<std::valarray<T>> const &a2)
{
    std::valarray<std::valarray<T>> res(a2.size());
    
    #pragma omp parallel for default(none) shared(res,a1,a2) schedule(dynamic)
    for(size_t i = 0; i < a2.size(); ++i)
    {
        res[i] = a1 * a2[i];
    }

    return res;
}

template <typename T>
std::valarray<std::valarray<T>> operator+(std::valarray<std::valarray<T>> const &a1, std::valarray<std::valarray<T>> const &a2)
{
    std::valarray<std::valarray<T>> res(a2.size());
    
    #pragma omp parallel for default(none) shared(res,a1,a2) schedule(dynamic)
    for(size_t i = 0; i < a2.size(); ++i)
    {
        res[i] = a1[i] + a2[i];
    }

    return res;
}

template <typename T>
T norm2(std::valarray<T> const &array)
{
    // compute and return the norm2 of a valarray
    return std::sqrt((array * array).sum());
}

double norm2(std::valarray<std::valarray<double>> const &array)
{
    // compute and return the norm2 of a valarray of valarray
    double nres(0.e0);   
    
    #pragma omp parallel for default(none) shared(array) reduction(+:nres) schedule(dynamic)
    for(size_t i = 0;i < array.size();++i){
        nres += std::pow(norm2(array[i]), 2);
    }
    
    return std::sqrt(nres);
}

double scalarproduct(std::valarray<std::valarray<double>> const &a1, std::valarray<std::valarray<double>> const &a2)
{
    // compute and return the norm2 of a valarray of valarray
    double res(0.e0);
    if (a1.size() != a2.size() || a1[0].size() != a2[0].size())
    {
        std::cerr << "Error: the two valarray of valarray must have the same size" << std::endl;
        exit(1);
    }
    
    for (size_t i = 0; i < a1.size(); i++)
    {
        res += (a1[i] * a2[i]).sum();
    }

    return res;
}

template <typename T>
void vvadd(std::valarray<std::valarray<T> > &inout, std::valarray<std::valarray<T> > &in) {
    inout = inout + in;
}

template <typename T>
void vadd(std::valarray<T> &inout, std::valarray<T> &in) {
    inout = inout + in;
}

#pragma omp declare reduction(vd_plus : std::valarray<double> : \
                              vadd(omp_out,omp_in)) \
                    initializer(omp_priv = std::valarray<double>(omp_orig.size()))

#pragma omp declare reduction(vi_plus : std::valarray<int> : \
                              vadd(omp_out,omp_in)) \
                    initializer(omp_priv = std::valarray<int>(omp_orig.size()))

#pragma omp declare reduction(vvd_plus : std::valarray<std::valarray<double> > : \
                              vvadd(omp_out,omp_in)) \
                    initializer(omp_priv = std::valarray<std::valarray<double> > (std::valarray<double>(omp_orig[0].size()),omp_orig.size()))

#pragma omp declare reduction(vvi_plus : std::valarray<std::valarray<int> > : \
                              vvadd(omp_out,omp_in)) \
                    initializer(omp_priv = std::valarray<std::valarray<int> > (std::valarray<int>(omp_orig[0].size()),omp_orig.size()))


// progress bar
void progress(double const& progress, double const& nsteps, int const cTotalLength = 10)
{
    std::ostringstream foo;
    double lProgress(progress/nsteps);
    if(nsteps>10){
        lProgress = progress/(std::floor(nsteps/10.0)*10.0);
    }
    if(std::abs(std::remainder(lProgress,0.1))>=1/(nsteps+1)){
        return;
    }
    if(lProgress > 1.0 || lProgress < 0) return;
    int c1(std::floor(lProgress * cTotalLength));
    int c2(cTotalLength - c1);

    foo << "[" <<                                           //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
        std::string(c1, 'X') <<       // printing filled part
        std::string(c2, '-') << // printing empty part
        "] " << std::setprecision(3) << 100 * lProgress << "%";
    std::cout << foo.str() << std::endl;
}

// histogram
std::valarray<int> histogram(std::valarray<double> const &data, double const &Lmin, double const &Lmax, size_t const &Nbins)
{
    // compute the histogram of the data
    std::valarray<int> histo(Nbins); 

    #pragma omp parallel for default(none) shared(data,Lmin,Lmax,Nbins) reduction(vi_plus:histo) schedule(dynamic)
    for(size_t i = 0;i < data.size();++i){
        double ind(std::floor(Nbins * ((data[i] - Lmin) / (Lmax - Lmin))));
        if (ind >= 0 && ind < Nbins)
        {
            histo[ind]++;
        }
    }

    return histo;
}

// Engine class
Engine::Engine(ConfigFile const &configFile)
    : last(0), t(0.e0), cstep(0)
{
    N = configFile.get<unsigned int>("N");
    position = std::valarray<std::valarray<double>>(std::valarray<double>(3), N);
    velocity = std::valarray<std::valarray<double>>(std::valarray<double>(3), N);
    initialize(configFile.get<std::string>("input"));
    posshift = std::valarray<std::valarray<double>>(std::valarray<double>(3), N);
    //it is redone in the steps for 1/2 advance in vv but here we let it for other implementations
    initpos = position;
    initvel = velocity;

    rcutoff = configFile.get<double>("rcutoff");
    Nbins = configFile.get<unsigned int>("Nbins");
    Nsteps = configFile.get<unsigned int>("Nsteps");
    histlength = configFile.get<double>("histlength");

    nbofr = std::valarray<std::valarray<int>>(std::valarray<int>(Nbins), Nsteps);
    pv2 = std::valarray<std::valarray<int>>(std::valarray<int>(Nbins), Nsteps);
    pv = std::valarray<std::valarray<int>>(std::valarray<int>(Nbins), Nsteps);
    vacf = std::valarray<double>(Nsteps);
    msd = std::valarray<double>(Nsteps);
    enpot = std::valarray<double> (Nsteps);

    // Simulation parameters
    dt = configFile.get<double>("dt");
    tend = dt * (Nsteps - 0.01);
    sampling = configFile.get<unsigned int>("sampling");
    L = configFile.get<double>("L");
    Lbin = 0.5 * L / Nbins;

    // Opening the output file
    printoption = configFile.get<unsigned int>("printoption");
    outputFilename = configFile.get<std::string>("output");
    precision = configFile.get<unsigned int>("precision");
    if (printoption > 0)
    {
        outputFile = new std::ofstream((outputFilename + ".out").c_str());
        outputFile->precision(precision);
    }
    else
    {
        outputFile = nullptr;
    }
};

Engine::~Engine()
{
    if (outputFile != nullptr)
    {
        outputFile->close();
    }
    delete outputFile;
};

void Engine::initialize(std::string const &filename)
{
    std::ifstream inputFile(filename.c_str(), std::ifstream::in);
    if (inputFile.is_open())
    {
        std::string line;
        size_t i(0);
        while (std::getline(inputFile, line))
        {
            std::istringstream iss(line);
            std::valarray<double> pos(3);
            iss >> pos[0] >> pos[1] >> pos[2];
            if (i < N)
            {
                position[i] = pos;
            }
            else
            {
                velocity[i % N] = pos;
            }
            ++i;
        };
        inputFile.close();
    }
    else
    {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(1);
    }
}

void Engine::run()
{
    printOut(true); // Write the initial conditions
    enpot[0] = 0.e0;
    while (cstep < Nsteps)
    {
        step(dt);        // Actualize the system
        histo(histlength);
        t += dt;
        printOut(false); // Write the actual results
        progress(cstep, Nsteps);
        ++cstep;
    }
    progress(Nsteps, Nsteps);
    printOut(true); // Write the last step

    // Print the final position and velocity
    std::ofstream outposvel(outputFilename + ".final", std::ofstream::out);
    outposvel.precision(precision);
    printState(outposvel);
    outposvel.close();

    if (printoption == 2)
    {
        printFile(outputFilename + ".analyse");
    }
};

void Engine::printState(std::ostream &os)
{
    std::string out;
    for (size_t i(0); i < position.size(); ++i)
    {
        os << position[i][0] << " " << position[i][1] << " " << position[i][2] << std::endl;
        out += std::to_string(velocity[i][0]) + " " + std::to_string(velocity[i][1]) + " " + std::to_string(velocity[i][2]) + "\n";
    }
    os << out;
}

void Engine::printOut(bool write)
{
    if (printoption == 0)
    {
        return;
    }
    // Writing every [sampling] steps, or if write is true
    if ((!write && last >= sampling) || (write && last != 1))
    {
        *outputFile << t << " " << printSystem() << std::endl;
        /*over this it's written with *outputFile << */
        last = 1;
    }
    else
    {
        last++;
    }
}

void Engine::boundary(std::valarray<double> &dr_)
{
    for(auto &dri : dr_)
    {
        dri -= L * std::round(dri / L);
    }
}

void Engine::boundary()
{
    // position rescaling
    #pragma omp parallel for default(none) shared(position, posshift, L) schedule(dynamic)
    for(size_t i = 0;i<position.size();++i){
        for(size_t j = 0;j<position[0].size();++j){
            double sh(std::floor(position[i][j]/L));
            posshift[i][j] += sh*L;
            position[i][j] -= sh*L;
        }
    }
}

void Engine::dynamiccoefs(){
    vacf[cstep] = scalarproduct(velocity, initvel) / 3 / N;      // np.mean([np.dot(b, a)/3. for a, b in zip(vel_init, vel)]);
    msd[cstep] = std::pow(norm2(initpos - position - posshift), 2) / N; // np.mean(((pos_init - (pos + L*shift))**2).sum(axis=1));
}

void Engine::histo(double const& lim){
    std::valarray<double> vx(velocity.size());
    std::valarray<double> v2(velocity.size());
    for (size_t i(0); i < velocity.size(); ++i)
    {
        vx[i] = velocity[i][0];
        v2[i] = norm2(velocity[i]);
    }
    pv2[cstep] = histogram(v2, 0, lim, Nbins);
    pv[cstep] = histogram(vx, -std::sqrt(lim), std::sqrt(lim), Nbins);
}

std::string Engine::printSystem()
{
    // print the state of the system
    std::string state = std::to_string(enpot[cstep] / N) + " " + std::to_string(enkin() / N) + " " + std::to_string((enkin()+enpot[cstep]) / N);
    //std::string state = std::to_string(enpot[cstep] / N)+" "+std::to_string(Enpot() / N) + " " + std::to_string(enkin() / N);
    return state;
}

std::valarray<std::valarray<double>> Engine::f(std::valarray<std::valarray<double>> const &position_, std::valarray<std::valarray<double>> const &velocity_)
{
    std::valarray<std::valarray<double>> force({0, 0, 0}, position_.size());
    std::valarray<int> nbofrtmp(Nbins);
    double entmp(0.e0);

    #pragma omp parallel for default(none) shared(position_,velocity_) reduction(vi_plus:nbofrtmp) reduction(vvd_plus:force) reduction(+:entmp) schedule(dynamic)
    for (size_t i = 0; i < position_.size(); ++i){
        for (size_t j = i + 1; j < position_.size(); ++j){
            std::valarray<double> dr = position_[i] - position_[j];
            
            // boundary conditions
            boundary(dr);
            
            // compute the force
            double dr2 = std::pow(norm2(dr), 2);

            std::valarray<double> F(3);
            if (dr2 < rcutoff * rcutoff)
            {
                entmp += 4 * (std::pow(dr2, -6) - std::pow(dr2, -3));
                F = 4 * (12 * std::pow(dr2, -7) - 6 * std::pow(dr2, -4)) * dr;
            }
            
            force[i] += F;
            force[j] -= F;
            
            if (dr2 < std::pow(Nbins * Lbin, 2))
            {   
                // compute the histogram of the radial distribution function
                size_t bin(std::floor(std::sqrt(dr2) / Lbin));
                nbofrtmp[bin] += 1;
            }
        }
    }

    nbofr[cstep] = nbofrtmp;
    enpot[cstep] += entmp+8 * M_PI * (std::pow(rcutoff, -9) / 9. - std::pow(rcutoff, -3) / 3.); // add the tail of potential energy

    return force;
}

double Engine::Enpot()
{
    double resenpot(0.e0);
    
    #pragma omp parallel for default(none) shared(position,rcutoff) reduction(+:resenpot) schedule(dynamic)
    for (size_t i = 0; i < position.size(); ++i)
    {
        for (size_t j = i + 1; j < position.size(); ++j)
        {
            std::valarray<double> dr = position[i] - position[j];
            boundary(dr);
            double dr2 = std::pow(norm2(dr), 2);
            if (dr2 < rcutoff * rcutoff)
            {
                resenpot += 4 * (std::pow(dr2, -6) - std::pow(dr2, -3));
            }
        }
    }

    resenpot += 8 * M_PI * (std::pow(rcutoff, -9) / 9. - std::pow(rcutoff, -3) / 3.); // add the tail of potential energy
    return resenpot;
}

double Engine::enkin() const
{
    double enkin(0.e0);
    for (auto const &v : velocity)
    {
        enkin += 0.5 * std::pow(norm2(v), 2);
    }
    return enkin;
}

void Engine::printFile(std::string const& filename)
{
    std::ofstream output((outputFilename + ".gofr").c_str());
    std::valarray<std::valarray<double>> Gofr(gofr());
    for (size_t i(0); i < Nbins; ++i)
    {
        output << Gofr[0][i] << " " << Gofr[1][i] << std::endl;
    }
    output.close();

    output = std::ofstream((outputFilename + ".pv").c_str());
    for (size_t i(0); i < pv.size(); ++i)
    {
        for (size_t j(0); j < pv[0].size(); ++j)
        {
            output << pv[i][j] << " ";
        }
        output << std::endl;
    }
    output.close();

    output = std::ofstream((outputFilename + ".pv2").c_str());
    for (size_t i(0); i < pv2.size(); ++i)
    {
        for (size_t j(0); j < pv2[0].size(); ++j)
        {
            output << pv2[i][j] << " ";
        }
        output << std::endl;
    }
    output.close();

    output = std::ofstream((outputFilename + ".msd-vacf").c_str());
    for (size_t i(0); i < msd.size(); ++i)
    {
        output << msd[i] << " " << vacf[i] << std::endl;
    }
    output.close();
}

const std::valarray<std::valarray<double>> Engine::gofr()
{
    std::valarray<double> gofr_m(Nbins);
    std::valarray<double> r(Nbins);
    double V(std::pow(2 * Nbins * Lbin, 3));

    #pragma omp parallel for default(none) shared(nbofr,gofr_m)
    for (size_t i = 0; i < Nbins; ++i)
    {
        for (size_t j = 0; j < nbofr.size(); ++j)
        {
            gofr_m[i] += nbofr[j][i];
        }
        gofr_m[i] /= nbofr.size();
    }

    #pragma omp parallel for default(none) shared(r,gofr_m,Lbin,V,N)
    for (size_t i = 0; i < Nbins; ++i)
    {
        r[i] = Lbin * (i + 0.5);
        gofr_m[i] = 2 * V / (N * (N - 1)) / (4 * M_PI * r[i] * r[i] * Lbin) * gofr_m[i];
    }
    return std::valarray<std::valarray<double>>({r, gofr_m});
}

/* Subclasses of Engine */

// Velocity Verlet
EngineVelocityVerlet::EngineVelocityVerlet(ConfigFile const &configFile)
    : Engine(configFile)
{
    ft = f(position, velocity);
}

void EngineVelocityVerlet::step(double dt_)
{
    velocity += 0.5 * dt_ * ft;
    position += dt_ * velocity;

    if(t==0)
    {
        initpos = position;
        initvel = velocity;
    }

    boundary();
    dynamiccoefs();

    std::valarray<std::valarray<double>> nextf(f(position, velocity));
    velocity += 0.5 * dt_ * nextf;
    ft = nextf;
}

// Gear
EngineGear::EngineGear(ConfigFile const &configFile)
    : Engine(configFile)
{
    // Gear coefficients
    a = std::valarray<double>({1.0 / 6.0, 5.0 / 6.0, 1.0, 1.0 / 3.0});

    // Initialize y

    std::valarray<std::valarray<double>> f0 = f(position, velocity);
    double tmppot(enpot[0]);
    // Do one step with Euler
    std::valarray<std::valarray<double>> velocity_ = velocity + 0.5 * dt * f0;
    std::valarray<std::valarray<double>> position_ = position + dt * velocity;
    // fprime
    std::valarray<std::valarray<double>> f1 = f(position_, velocity_) - f0;
    enpot[0] = tmppot;

    y = std::valarray<std::valarray<std::valarray<double>>>({position, dt * velocity, dt * dt * 0.5 * f0, 1.0 / 6.0 * dt * dt * f1});
}

void EngineGear::step(double dt_)
{
    yp = std::valarray<std::valarray<std::valarray<double>>>({y[0] + y[1] + y[2] + y[3], y[1] + 2 * y[2] + 3 * y[3], y[2] + 3 * y[3], y[3]});
    std::valarray<std::valarray<double>> ftmp = f(yp[0], 1.0 / dt * y[1]);
    y = std::valarray<std::valarray<std::valarray<double>>>({   yp[0] + 0.5*a[0] * dt_ * dt_ * ftmp - 0.5*a[0] * dt_ * yp[2], 
                                                                yp[1] + 0.5*a[1] * dt_ * dt_ * ftmp - 0.5*a[1] * dt_ * yp[2], 
                                                                0.5*a[2] * dt_ * dt_ * ftmp,
                                                                yp[3] + 0.5*a[3] * dt_ * dt_ * ftmp - 0.5*a[3] * dt_ * yp[2]});
    
    velocity = 1.0 / dt_ * y[1];
    position = y[0];

    if(t==0)
    {
        initpos = y[0];
        initvel = velocity;
    }
    
    boundary();
    dynamiccoefs();
    y[0] = position;
}

/* Sub-subclasses of Engine */
// Velocity Verlet with Velocity Rescaling
EngineVVVR::EngineVVVR(ConfigFile const &configFile)
    : EngineVelocityVerlet(configFile)
{
    T = configFile.get<double>("T");
}

void EngineVVVR::step(double dt_)
{
    EngineVelocityVerlet::step(dt_);
    // Rescale the velocities
    velocity = std::sqrt(N * 3 * T / (2 * enkin())) * velocity;
}

// Velocity Verlet with Nose-Hoover
EngineVVNH::EngineVVNH(ConfigFile const &configFile)
    : EngineVelocityVerlet(configFile)
{
    T = configFile.get<double>("T");
    Q = configFile.get<double>("Q");
    xi = configFile.get<double>("xi");
    lns = configFile.get<double>("lns");
}

double EngineVVNH::dsumv2() const
{
    return (2 * enkin() - 3. * N * T) / Q;
}

double EngineVVNH::enNH() const
{
    return 0.5 * Q * xi * xi + 3. * N * T * lns;
}

void EngineVVNH::step(double dt_)
{
    velocity += 0.5 * dt_ * ft - 0.5 * dt_ * xi * velocity;
    position += dt_ * velocity;

    if(t==0)
    {
        initpos = position;
        initvel = velocity;
    }

    dynamiccoefs();

    //boundary conditions (which is for some reason not in md.py)
    boundary();

    double ds2(dsumv2());
    lns += xi * dt_ + 0.5 * ds2 * std::pow(dt_, 2);
    xi += 0.5 * ds2 * dt_;
    std::valarray<std::valarray<double>> nextf(f(position, velocity));
    velocity += 0.5 * dt_ * nextf - 0.5 * dt_ * xi * velocity;
    ft = nextf;
    xi += 0.5 * dsumv2() * dt;
}

std::string EngineVVNH::printSystem()
{
    std::string state;
    state += Engine::printSystem();
    state += " " + std::to_string(enNH() / N);
    state += " " + std::to_string(lns) + " " + std::to_string(xi);
    return state;
}

// Velocity Verlet with Berendsen
EngineVVB::EngineVVB(ConfigFile const &configFile)
    : EngineVelocityVerlet(configFile)
{
    T = configFile.get<double>("T");
    double tau(configFile.get<double>("tau"));
    gamma = 1.e0 / (2.e0 * tau);
}

std::valarray<std::valarray<double>> EngineVVB::f(std::valarray<std::valarray<double>> const &position_, std::valarray<std::valarray<double>> const &velocity_)
{
    return Engine::f(position_, velocity_) + gamma * (3 * N * T / (2 * enkin()) - 1) * velocity_;
}

// Gear with Velocity Rescaling
EngineGVR::EngineGVR(ConfigFile const &configFile)
    : EngineGear(configFile)
{
    T = configFile.get<double>("T");
}

void EngineGVR::step(double dt_)
{
    EngineGear::step(dt_);
    // Rescale the velocities
    velocity = std::sqrt(N * 3 * T / (2 * enkin())) * velocity;
    y[1] = dt*velocity;
}

// Gear with Nose-Hoover
EngineGNH::EngineGNH(ConfigFile const &configFile)
    : EngineGear(configFile)
{
    T = configFile.get<double>("T");
    Q = configFile.get<double>("Q");
    xi = configFile.get<double>("xi");
    lns = configFile.get<double>("lns");
    
    double e1(dsumv2());
    double tmp(enpot[0]);
    std::valarray<std::valarray<double>> f0 = f(position, velocity);
    std::valarray<std::valarray<double>> velocity_ = velocity + 0.5 * dt * f0;
    enpot[0] = tmp;
    double e2 = (norm2(velocity_) - 3. * N * T) / Q;

    nhy = std::valarray<double>({lns, dt*xi, 0.5*dt*dt*e1, dt*dt/6.0*e2-dt*dt/6.0*e1});
    nhyp = std::valarray<double>(4);
}

double EngineGNH::dsumv2() const
{
    return (2 * enkin() - 3. * N * T) / Q;
}

double EngineGNH::enNH() const
{
    return 0.5 * Q * xi * xi + 3. * N * T * lns;
}

void EngineGNH::step(double dt_)
{
    yp = std::valarray<std::valarray<std::valarray<double>>>({y[0] + y[1] + y[2] + y[3], y[1] + 2 * y[2] + 3 * y[3], y[2] + 3 * y[3], y[3]});
    nhyp = std::valarray<double>({nhy[0] + nhy[1] + nhy[2] + nhy[3], nhy[1] + 2 * nhy[2] + 3 * nhy[3], nhy[2] + 3 * nhy[3], nhy[3]});
    std::valarray<std::valarray<double>> ftmp = f(yp[0], 1.0 / dt * y[1]);
    y = std::valarray<std::valarray<std::valarray<double>>>({   yp[0] + 0.5*a[0] * dt_ * dt_ * ftmp - 0.5*a[0] * dt_ * yp[2], 
                                                                yp[1] + 0.5*a[1] * dt_ * dt_ * ftmp - 0.5*a[1] * dt_ * yp[2], 
                                                                0.5*a[2] * dt_ * dt_ * ftmp,
                                                                yp[3] + 0.5*a[3] * dt_ * dt_ * ftmp - 0.5*a[3] * dt_ * yp[2]});
    velocity = 1.0 / dt_ * y[1];
    double tmp(dsumv2());
    nhy = std::valarray<double>({   nhyp[0] + 0.5*a[0] * dt_ * dt_ * tmp - 0.5*a[0] * dt_ * nhyp[2], 
                                    nhyp[1] + 0.5*a[1] * dt_ * dt_ * tmp - 0.5*a[1] * dt_ * nhyp[2], 
                                    0.5*a[2] * dt_ * dt_ * tmp,
                                    nhyp[3] + 0.5*a[3] * dt_ * dt_ * tmp - 0.5*a[3] * dt_ * nhyp[2]});                                                   
    
    if(t==0)
    {
        initpos = y[0];
        initvel = velocity;
    }

    position = y[0];
    dynamiccoefs();
}

std::string EngineGNH::printSystem()
{
    std::string state;
    state += EngineGear::printSystem();
    state += " " + std::to_string(enNH() / N);
    state += " " + std::to_string(lns) + " " + std::to_string(xi);
    return state;
}

// Gear with Berendsen
EngineGB::EngineGB(ConfigFile const &configFile)
    : EngineGear(configFile)
{
    T = configFile.get<double>("T");
    double tau(configFile.get<double>("tau"));
    gamma = 1.e0 / (2.e0 * tau);
}

std::valarray<std::valarray<double>> EngineGB::f(std::valarray<std::valarray<double>> const &position_, std::valarray<std::valarray<double>> const &velocity_)
{
    return Engine::f(position_, velocity_) + gamma * (3 * N * T / (2 * enkin()) - 1) * velocity_;
}