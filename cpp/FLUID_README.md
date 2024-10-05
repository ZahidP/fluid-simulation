### General Scheme

#### Central Difference Approximation for Divergence

To enforce incompressibility in fluid dynamics, we approximate the divergence of the velocity field using a central difference scheme on a grid. For a 2D grid, the divergence at grid point \( (i, j) \) is:

$$
\mathbf{div}(\mathbf{u})_{i,j} = \frac{u_{i+1,j} - u_{i-1,j}}{2 \Delta x} + \frac{v_{i,j+1} - v_{i,j-1}}{2 \Delta y}
$$

Where:

- u and v are the velocity components in the x and y directions, respectively.
- delta_x and delta_y are the grid spacings in the x and y directions.


#### Using Divergence to Update Velocities

After calculating the divergence, it is used to update the velocity field to ensure incompressibility. This is typically done by solving for a pressure correction \( p \), and then using it to correct the velocities:

$$
\mathbf{u}_{n+1} = \mathbf{u}_n - \Delta t \cdot \nabla p
$$

Where:
- nabla(p) is the pressure gradient.
- The pressure \( p \) is obtained by solving a Poisson equation derived from the divergence of the velocity field.

This step ensures that the velocity field remains divergence-free, satisfying the incompressibility condition for fluid flow.

#### Velocity Advection

In fluid dynamics, advection is the process of transporting quantities (like velocity) across the grid. Using an Eulerian method, we typically employ **semi-Lagrangian advection** to update the velocity at a grid point by tracing backward along the velocity field to find the origin of the velocity being advected. The advection step can be represented as:

$$
\mathbf{u}_{\text{advected}}(\mathbf{x}) = \mathbf{u}(\mathbf{x} - \Delta t \cdot \mathbf{u}(\mathbf{x}))
$$

Where:
- **u_advected(x)** is the new velocity at the grid point **x**.
- **u(x)** is the velocity at grid point **x** before advection.
- **Δt** is the time step size.
- **x - Δt * u(x)** represents the location from which the velocity is advected, found by tracing backward along the velocity field.


### Differences in Schemes

#### Pressure Calculation

#### Scheme 1 (Pressure Correction)

Pressure correction is applied directly using the divergence and fluid neighbors:

$$
P_{\text{correction}} = -\frac{\text{div}(\mathbf{u})}{n_{\text{fluid neighbors}}}
$$

#### Scheme 2 (Poisson Pressure Equation)

Based on the pressures of neighboring cells:

$$
P_{i,j}^{\text{new}} = P_{i,j-1} + P_{i,j+1} + P_{i-1,j} + P_{i+1,j} - \text{div}(\mathbf{u}) \cdot \Delta x^2 \cdot \rho
$$

#### Velocity Update

#### Scheme 1 (Direct Velocity Update from Pressure Correction)

Velocity updates are applied directly to neighboring velocities based on the pressure correction.

For \( U \) (horizontal velocity):

$$
U_{i,j} += \text{fluid}_{\text{right}} \cdot P_{\text{correction}}, \quad U_{i,j-1} -= \text{fluid}_{\text{left}} \cdot P_{\text{correction}}
$$

For \( V \) (vertical velocity):

$$
V_{i,j} += \text{fluid}_{\text{down}} \cdot P_{\text{correction}}, \quad V_{i-1,j} -= \text{fluid}_{\text{up}} \cdot P_{\text{correction}}
$$

#### Scheme 2 (Velocity Update from Pressure Gradient)

The velocities are updated based on the pressure gradient after the pressure has been solved using the Poisson equation.

For \( U \) (horizontal velocity):

$$
U_{i,j} -= \frac{P_{i,j+1} - P_{i,j}}{\Delta x} \cdot \frac{\Delta t \cdot \text{fluid}_{i,j}}{\rho}
$$

For \( V \) (vertical velocity):

$$
V_{i,j} -= \frac{P_{i+1,j} - P_{i,j}}{\Delta x} \cdot \frac{\Delta t \cdot \text{fluid}_{i,j}}{\rho}
$$