# A simple parallel solution method for the Navier-Stokes Cahn-Hilliard equations
This repository provides the implementation of the model presented in the publication 'A simple parallel solution method for the Navier-Stokes Cahn-Hilliard equations'.

### Files
* `main.cc` contains core code of the implementation of the numerical model
* `helpers.h` handles output into VTK Files and data structure
* `config.conf` contains all constants and simulation parameters

### Compilation
You can compile the code using the following command:

```
g++ -std=c++17 -fopenmp -O3 main.cc
```

### Simulation output
The application needs a subfolder named `paraview`. Into this directory the simulation will write its output files which can be visualized in [ParaView](www.paraview.org). Additionally, a csv-file named `benchmark.csv` will be created. It contains the calculated benchmark quantities, which are described more in detail in the publication.