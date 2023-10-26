# PINN-2dNSEs
PINN for 2d NSEs test with lid driven cavity

![Lid-Driven](./image/Lid-Driven.gif)

\[
\rho \left ( \partial_{t} \mathbf{v} +  \left ( \mathbf{v}  \cdot \bigtriangledown  \right ) \mathbf{v}  \right ) - \eta \bigtriangleup \mathbf{v} + \bigtriangledown \mathbf{p} = \mathbf{f} 
\]
in $ \Omega $ $ \times $ $\left ( 0, T \right )$,
\[
                          div \mathbf{v}  = 0     
\]
in $ \Omega $ $ \times $ $\left ( 0, T \right ),$

$\mathbf{v} = \left ( u , v \right ) $ is the velocity vector. $\mathrm{}{p} $ is the pressure. $\mathrm{\rho} $ is the density and $\mathrm{\eta}$ is the viscosity.


**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$