# fys4411-project1
Project 1 for Computational physics 2 - FYS4411

This project have been done by Ali Reza Asghari Zadeh and Lars Opgård. The structure is based one this [template](https://github.com/larsfop/fys4411-project1/blob/main/README.md)

In this project we have created a Variational Monte Carlo program to compute the lowest state energy for a given trial wavefunction with both a non-interacting Harmoinc oscillator and a two-body interacting wavefunction. We have then introduced a brute-force Metropolis sampling algorithm which we upgraded to the Metropolis-Hastings method. We started by simply varying the wavefunction parameters, before we introduced a better optimization algorithm with the Stochastic gradient descent.


## Compiling and running the project
The recommend way to compile this project is by using CMake to create a Makefile that you can then run. You can install CMake through one of the Linux package managers, e.g., `apt install cmake`, `pacman -S cmake`, etc. For Mac you can install using `brew install cmake`. Other ways of installing are shown here: [https://cmake.org/install/](https://cmake.org/install/).

### Compiling the project using CMake
In a Linux/Mac terminal this can be done by the following commands
```bash
# Create build-directory
mkdir build

# Move into the build-directory
cd build

# Run CMake to create a Makefile
cmake ../ -DCMAKE_BUILD_TYPE=Release

# Make the Makefile using two threads
make -j2

# Move the executable to the top-directory
mv vmc ..
```
Or, simply run the script `compile_project` via
```bash
./compile_project
```
and the same set of commands are done for you. Now the project can be run by executing
```bash
./vmc
```
in the top-directory. The vmc program can take a few kwargs directly into the console, the keyword you can use is shown in the config file, while the types can be seen from the main file. To run the program a version of [openMP](https://www.openmp.org/) and [armadillo](https://arma.sourceforge.net/) is needed.

For the project we have had to use different variables set in the config. For filenames, all the used files are given with the repository, where the names of the files have these meanings
- "SG" means simplge gaussian
- "IW" means interacting wavefunction
- "SM" means brute-force Metropolis
- "HM" means Metropolis-Hastings
- "OP" means using gradient descent
- "#D" means # of dimensions
- "#P" means number of particles
- "VMC" just means variational Monte Carlo

The different problems were ran with different initial settings either set in the config or directly through the console. Differing from the config as given
### Simple Gaussian
- seed=125
- Particles varies from 1, 10, 100 and 500
- Dimensions vary from 1,2 and 3
- alpha=0.1
- beta=1.0
- steplenght=1 for the brute-force method (important)
- threadsused=1
Then the remaining differs based one what result you want

### Interacting Wavefunction
- seed=2024 and 1234, with 2024 being used for the final cycle and 1234 for the optimization
  This is done due two weird interaction with the threads for certain rng's
- Particles varies between 10, 50 and 100
With the rest depending on the results wanted


#### Cleaning the directory
Run `make clean` in the top-directory to remove the executable `vmc` and the `build`-directory.

#### Windows
Compilation of the project using Windows should work using CMake as it is OS-independent, but `make` does not work on Windows so the `compile_project`-script will not work.

