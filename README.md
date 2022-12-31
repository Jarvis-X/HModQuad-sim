# HModQuad-sim
Simulations for various structures in CoppeliaSim with Python API
## Dependencies
* numpy
* CoppeliaSim Edu

## The API files are included, which can be found at https://www.coppeliarobotics.com/
* remoteApi.dll
* sim.py
* simConst.py

## There are three simulation environments, each associated with one Python script
### Initiate the simulation environment, then run the python file. The robot is going t follow a trajectory
* plus.ttt -> plus.py
* 3x3mixture.ttt -> 3x3mixture.py
* 4x4TModules.ttt -> 4x4TModules.py
* ***NEW!*** 4x4TModulesNonRigidConnection.ttt -> 4x4TModules.py 
  * Instead of attaching the rotors to the multi-rotor rigid body, the modules are now connected to the multi-rotor rigid body in a modular way through dummies and links, allowing to emulate the bending of the connection and the misalignment between the modules.
* genConfig: **NEW** the script to generate structure configurations with homogeneous modules that satisfy 
a set of task requirements.
