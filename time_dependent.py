import tensorflow as tf
import matplotlib.pyplot as plt
from optimizer import L_BFGS_B
# tf.random.set_seed(1234)
import matplotlib.animation as animation
from data import Data
import data as dt

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


def InitializeModel(num_hidden_layers=5, num_neurons_per_layer=8):
    """
    Initialize the DNN model

    Args:
        num_hidden_layers: number of hidden layers
        num_neurons_per_layer : number of neurons each layer

    Returns:
        model: return the DNN model (the output refers to the psi and pressure, the gradient of psi represents velocity)
    """

    model = tf.keras.Sequential()  # Initialize a feedforward neural network
    model.add(tf.keras.Input(3))  # Input is three-dimensional

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation=tf.keras.activations.get('tanh'),
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    model.add(tf.keras.layers.Dense(2))  # Output is two-dimensional

    return model


class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the time dependent Navier-Stokes equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """
        self.model = model
        super().__init__(**kwargs)

    def call(self, xyt):
        """
        Computing derivatives for the time dependent Navier-Stokes equation.

        Args:
            xyt: input variable.

        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """
        x, y, t = [xyt[..., i, tf.newaxis] for i in range(xyt.shape[-1])]
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(x)
            ggg.watch(y)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                gg.watch(y)
                gg.watch(t)
                with tf.GradientTape(persistent=True) as g:
                    g.watch(x)
                    g.watch(y)
                    psi_p = self.model(tf.concat([x, y, t], axis=-1))
                    psi = psi_p[..., 0, tf.newaxis]
                    p = psi_p[..., 1, tf.newaxis]
                u = g.batch_jacobian(psi, y)[..., 0]
                v = -g.batch_jacobian(psi, x)[..., 0]
                p_x = g.batch_jacobian(p, x)[..., 0]
                p_y = g.batch_jacobian(p, y)[..., 0]
                del g
            u_t = gg.batch_jacobian(u, t)[..., 0]
            v_t = gg.batch_jacobian(v, t)[..., 0]
            u_x = gg.batch_jacobian(u, x)[..., 0]
            u_y = gg.batch_jacobian(u, y)[..., 0]
            v_x = gg.batch_jacobian(v, x)[..., 0]
            v_y = gg.batch_jacobian(v, y)[..., 0]
            del gg
        u_xx = ggg.batch_jacobian(u_x, x)[..., 0]
        u_yy = ggg.batch_jacobian(u_y, y)[..., 0]
        v_xx = ggg.batch_jacobian(v_x, x)[..., 0]
        v_yy = ggg.batch_jacobian(v_y, y)[..., 0]
        del ggg

        p_grads = p, p_x, p_y
        u_grads = u, u_x, u_y, u_xx, u_yy
        v_grads = v, v_x, v_y, v_xx, v_yy
        d_time = u_t, v_t

        return psi, p_grads, u_grads, v_grads, d_time


class PINN:
    """
    Build a physics informed neural network (PINN) model for the time dependent Navier-Stokes equation.

    Attributes:
        network: keras network model with input (x, y, t) and output (psi, p).
        rho: density.
        nu: viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, rho=1, nu=0.01):
        """
        Args:
            network: keras network model with input (x, y, t) and output (psi, p).
            RHO: density.
            nu: viscosity.
        """

        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the time dependent Navier-Stokes equation.

        Returns:
            PINN model for the time dependent Navier-Stokes equation with
                input:  [ (x, y, t) relative to equation,
                        (x, y, t) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero), as we assume a scenario with negligible external
                        forces, such as gravity. Please note that in our code, (u, v) denotes the u and v directional
                        equations, not just direct velocity representations.
                        (f_div, f_div) relative to equation (must be zero), divergence free condition.
                        (psi, psi) relative to boundary condition (psi is duplicated because outputs require the same
                        dimensions), (u, v) relative to boundary condition ]
        """

        # equation input: (x, y, t)
        xyt_eqn = tf.keras.layers.Input(shape=(3,))
        # boundary condition
        xyt_bnd = tf.keras.layers.Input(shape=(3,))

        # compute gradients relative to equation
        _, p_grads, u_grads, v_grads, d_time = self.grads(xyt_eqn)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads
        u_t, v_t = d_time
        # compute equation loss
        u_eqn = u * u_x + v * u_y + u_t + p_y / self.rho - self.nu * (u_xx + u_yy)
        v_eqn = u * v_x + v * v_y + v_t + p_x / self.rho - self.nu * (v_xx + v_yy)

        # divergence free condition
        f_div = u_y + v_x

        uv_eqn = tf.concat([v_eqn, u_eqn], axis=-1)
        div_e = tf.concat([f_div, f_div], axis=-1)

        # compute gradients relative to boundary condition
        psi_bnd, _, u_bnd, v_bnd, d_time = self.grads(xyt_bnd)
        # those two are velocity representation
        u_bnd = u_bnd[0]
        v_bnd = v_bnd[0]
        # compute boundary condition loss
        psi_bnd = tf.concat([psi_bnd, psi_bnd], axis=-1)
        uv_bnd = tf.concat([u_bnd, v_bnd], axis=-1)

        # build PINN model for the time dependent Navier-Stokes equation
        return tf.keras.models.Model(
            inputs=[xyt_eqn, xyt_bnd], outputs=[uv_eqn, div_e, psi_bnd, uv_bnd])


def plot_loss(loss, name: str):
    plt.plot([i for i in range(len(loss))], loss)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.savefig('../image/' + name + '_loss.png')


def animate(i):
    # clear present plot on the axis to show animation
    ax[0].clear()
    ax[1].clear()
    global cb1
    if cb1:
        cb1.remove()
    _u, _v = data_u[i]
    x, y = coordinates
    ax[0].streamplot(x, y, _u, _v, cmap='plasma')
    ax1 = ax[1].contourf(x, y, data_psi[i], cmap='plasma')
    cb1 = plt.colorbar(ax1, ax=ax[1], shrink=0.65)
    # cb1.set_ticks([(i - 3) / 10 for i in range(7)])
    # aspect ratio of plot is preserved
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    # set title for subpltos
    ax[0].set_title('Velocity Field')
    ax[1].set_title('Pressure Field')


if __name__ == '__main__':
    TRAIN = True
    name = 'pipe'
    if TRAIN:  # train model
        # build a core network model
        network = InitializeModel()
        network.summary()
        # build a PINN model
        pinn = PINN(network, rho=dt.RHO, nu=dt.NU).build()
        x_train, y_train, name = Data.pipe()
        # train the model using L-BFGS-B algorithm
        lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, maxiter=dt.MAX_ITER)
        lbfgs.fit()
        loss = lbfgs.logger
        # save loss and model
        # plot_loss(loss, f'{name}')
        tf.keras.models.save_model(network, f'./{name}')

    # else:  # test model
        data_u, data_psi, coordinates, name = Data.test_model(name)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'{name}')
        # set the distance of two plot
        plt.subplots_adjust(wspace=0.4)
        # Call animate method
        cb1 = None  # colorbar
        ani = animation.FuncAnimation(fig, animate, dt.NUMBER_OF_FRAMES, interval=50, blit=False)
        anis = animation.FFMpegWriter(fps=10)

        # save v, p animation
        # ani.save(f'../image/{name}.gif', writer='pillow', fps=10)

        # Display the plot
        plt.show()
