# Fluid Simulation

There are two parts to this project. 

1. C++ fluid simulation and data generation.

2. Python neural network trainer and simulation.

The basic idea is to run the C++ fluid simulation, then use that data to train the neural network models. The models can be used to run a neural network driven simulation of fluid mechanics.

*I initially had to run this on Mac MPS hardware so I ended up using small data.*

#### Outputs

See `assets` for GIFs of the simulations. 

- *Note*: I'm not sure why the cpp gif renders so slowly but if you run the simulation it should render faster.

## For OMP Parallelism in XCode

Set these flags:

- -fopenmp
- -lomp

## C++ Simulator

- Run the main.cpp to generate the data. 
- You can update variables within the file to make adjustments.
- SFML is a requirement for this project
- There is an environment variable that is required to be set for csv and png outputs.

### General Scheme

- Divide into grid
- Central difference approximation for divergence/pressure correction
- Use pressure correction to update velocities
- Advect velocities


#### MacMethod.cpp/PressureCorrection.cpp

These ended up converging to very similar code so they produce similar results.

#### PressureFree.cpp

This runs a fluid simulation without direct use of pressure to drive the particle movement.
See here: https://www.youtube.com/watch?v=iKAVRgIrUOU&t=378s

See differences between schemes in `cpp/FLUID_README.MD`


## Python Training & Simulation

### GNN

Uses a standard GNN with neighborhood aggregation scheme. 

The graph structure is built using the code seen in `utils/create_grid_graph_with_angles.py`.
You can add or remove neighbors in that function (see commented code).

### GAT

The standard GNN neighbor aggregation scheme cannot properly adjust messages based on other source messages in the neighborhood. When performing inference, source nodes produce messages for a target node, independent of other source nodes. The graph attention network *"ties"* source nodes together during the aggregation scheme, such that they can inform the final aggregation in the target node.

### CNN

Naturally, a grid structure with several channels would seemingly to work well with a CNN. Whether this works for a fluid simulation is another question.

### Custom Loss Function

I decided to include a loss function that would force incompressibility such that the sum of a central cell plus its 4 direct neighbors should sum to 0 in the U and V channels. 

## Relevant Documentation

### Iterative Pressure Projection Methods
http://mseas.mit.edu/publications/PDF/Aoussou_et_al_JCP2018.pdf

### Pressure Projection
https://en.wikipedia.org/wiki/Projection_method_(fluid_dynamics)
https://math.berkeley.edu/~chorin/chorin67.pdf 

### Pressure Correction
https://en.wikipedia.org/wiki/Pressure-correction_method
http://mseas.mit.edu/publications/Theses/aoussou_sm_thesis_2016_06.pdf

### Marker and Cell
https://web.stanford.edu/class/cs205b/lectures/lecture17.pdf


### Eulerian Fluid Simulation
https://www.youtube.com/watch?v=iKAVRgIrUOU&t=378s

### FLIP
https://www.youtube.com/watch?v=XmzBREkK8kY&t=229s

