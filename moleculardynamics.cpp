#include <iostream>
#include <string>
#include "ConfigFile.h"   // use to read the configuration file
#include "Engine.h"       // use to run the simulation

using namespace std;  

int main(int argc, char* argv[])
{
  string inputPath("configuration.in");
  if(argc>1)  // Input file specified by the user
    inputPath = argv[1];

  ConfigFile configFile(inputPath); // Parameters are stocked in a map

  for(int i(2); i<argc; ++i) // Complementary inputs ("./MD configuration.in input_scan=[value]")
    configFile.process(argv[i]);

  Engine* engine(nullptr);
  string schema(configFile.get<string>("method"));
  string thermostat(configFile.get<string>("thermostat"));

  /* choose the right numerical methods */
  if(schema == "Velocity-Verlet" || schema == "VV")
  {
    if(thermostat == "NVE"){
      engine = new EngineVelocityVerlet(configFile);
    } else if (thermostat == "Nose-Hoover" || thermostat == "NH"){
      engine = new EngineVVNH(configFile);
    } else if (thermostat == "Velocity-Rescaling" || thermostat == "VR"){
      engine = new EngineVVVR(configFile);
    } else if (thermostat == "Berendsen" || thermostat == "B"){
      engine = new EngineVVB(configFile);
    } else {
      cerr << "Unknown thermostat" << endl;
      return -1;
    }
  } else if (schema == "Gear" || schema == "G"){
    if(thermostat == "NVE"){
      engine = new EngineGear(configFile);
    } else if (thermostat == "Nose-Hoover" || thermostat == "NH"){
      engine = new EngineGNH(configFile);
    } else if (thermostat == "Velocity-Rescaling" || thermostat == "VR"){
      engine = new EngineGVR(configFile);
    } else if (thermostat == "Berendsen" || thermostat == "B"){
      engine = new EngineGB(configFile);
    } else {
      cerr << "Unknown thermostat" << endl;
      return -1;
    }
  } else {
    cerr << "Unknown method" << endl;
    return -1;
  }
  cout << "Start of the simulation" << endl;

  engine->run(); // Execute simulation
    
  delete engine;
  
  cout << "End of the simulation." << endl;
  return 0;
}
