
# PINN-2dNSEs
PINN for 2d NSEs test with lid driven cavity

***Navier Stokes Equations***

$$ \rho \left ( \partial_{t} \mathbf{v} +  \left ( \mathbf{v}  \cdot \bigtriangledown  \right ) \mathbf{v}  \right ) - \eta \bigtriangleup \mathbf{v} + \bigtriangledown \mathbf{p} = \mathbf{f} $$

in $\Omega$ $\times$ $\left ( 0, T \right )$,

$$ div \mathbf{v}  = 0 $$


in $ \Omega $ $ \times $ $ \left ( 0, T \right ), $

$ \mathbf{v} = \left ( u , v \right )  $ is the velocity vector. $ \mathrm{}{p}  $ is the pressure. $ \mathrm{\rho}  $ is the density and $ \mathrm{\eta} $ is the viscosity.

***Loss functions***



![Lid-Driven](./image/Lid-Driven.gif)