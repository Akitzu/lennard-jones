/* The Engine class is the main motor of the program. It is used to read the
   configuration file, to simulate the system and write out the results.
*/

#ifndef ENGINE_H
#define ENGINE_H

#include "ConfigFile.h"
#include <map>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <valarray>

class Engine
{
    public:
        Engine(ConfigFile const& configFile); // Constructor
        virtual ~Engine();  // Virtual destructor
        void run();         // Run the simulation
    protected:
        unsigned int cstep;     // current time step
        unsigned int Nbins;     // number of bins for the radial distribution function
        double Lbin;            // bin size for the radial distribution function
        double histlength;      // length of the histogram v^2
        unsigned int N;         // number of particles
        unsigned int Nsteps;    // total number of time steps  
        double t;
        double dt;
        double tend;
        double L;               // box size
        double rcutoff;
        std::valarray<double> enpot;
        std::valarray<std::valarray<int> > nbofr;   
        std::valarray<std::valarray<int> > pv;
        std::valarray<std::valarray<int> > pv2;
        std::valarray<double> msd;
        std::valarray<double> vacf;
        std::valarray<std::valarray<double> > initpos;
        std::valarray<std::valarray<double> > initvel;
        std::valarray<std::valarray<double> > posshift;
        std::valarray<std::valarray<double> > position;
        std::valarray<std::valarray<double> > velocity;
        std::string outputFilename;
        double Enpot();
        virtual std::valarray<std::valarray<double> > f(std::valarray<std::valarray<double> > const& position_, std::valarray<std::valarray<double> > const& velocity_);      // compute the forces
        const std::valarray<std::valarray<double> > gofr();     // compute the radial distribution function
        double enkin() const;
        void boundary(std::valarray<double>& dr_);
        void boundary();
        void dynamiccoefs();
        void histo(double const& lim);
        virtual void printState(std::ostream& os);  // print the current position and velocity of the system
        virtual std::string printSystem();          // print the current energy and other parameters
    private:
        unsigned int sampling;  // number of time steps between each write of diagnostics
        unsigned int last;      // number of time steps since the last write of diagnostics
        unsigned int printoption;   // print option
        unsigned int precision;     // precision of the outputs
        std::ofstream *outputFile;  // pointer to the output file
        void initialize(std::string const& filename);  // initialize the positions and velocities
        void printOut(bool write);
        void printFile(std::string const& filename);
        virtual void step(double dt_) = 0;
};

// subclasses of Engine : different integration methods

class EngineVelocityVerlet : public Engine
{
    public:
        EngineVelocityVerlet(ConfigFile const& configFile);
    protected:
        virtual void step(double dt_) override;
        std::valarray<std::valarray<double> > ft;
};

class EngineGear : public Engine
{
    public:
        EngineGear(ConfigFile const& configFile);
    protected:
        virtual void step(double dt_) override;
        std::valarray<double> a;
        std::valarray<std::valarray<std::valarray<double> > > y;
        std::valarray<std::valarray<std::valarray<double> > > yp;
};

// sub-subclasses of Engine : different thermostats

// for Velocity Verlet
class EngineVVVR : public EngineVelocityVerlet
{
    public:
        EngineVVVR(ConfigFile const& configFile);
    protected:
        virtual void step(double dt_) override;
    private:
        double T;
};

class EngineVVNH : public EngineVelocityVerlet
{
    public:
        EngineVVNH(ConfigFile const& configFile);
        double enNH() const;
        double dsumv2() const;
    protected:
        virtual void step(double dt_) override;
    private:
        double T;
        double Q; 
        double xi; 
        double lns;
        virtual std::string printSystem() override;
};

class EngineVVB : public EngineVelocityVerlet
{
    public:
        EngineVVB(ConfigFile const& configFile);
    protected:
        virtual std::valarray<std::valarray<double> > f(std::valarray<std::valarray<double> > const& position_, std::valarray<std::valarray<double> > const& velocity_) override;      // compute the forces
    private:
        double T;
        double gamma;    
};


// for Gear
class EngineGVR : public EngineGear
{
    public:
        EngineGVR(ConfigFile const& configFile);
    protected:
        virtual void step(double dt_) override;
    private:
        double T;
};

class EngineGNH : public EngineGear
{
    public:
        EngineGNH(ConfigFile const& configFile);
        double enNH() const;
        double dsumv2() const; 
    protected:
        virtual void step(double dt_) override;
    private:
        double T;
        double Q; 
        double xi; 
        double lns;
        std::valarray<double> nhy;
        std::valarray<double> nhyp;
        virtual std::string printSystem() override;
};

class EngineGB : public EngineGear
{
    public:
        EngineGB(ConfigFile const& configFile);
    protected:
        virtual std::valarray<std::valarray<double> > f(std::valarray<std::valarray<double> > const& position_, std::valarray<std::valarray<double> > const& velocity_) override;      // compute the forces
    private:
        double T;
        double gamma;
};

#include "Engine.hpp"

#endif