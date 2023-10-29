import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from optimizer import L_BFGS_B
from matplotlib.colors import Normalize

tf.random.set_seed(1234)
import matplotlib.animation as animation

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


def InitializeModel(num_hidden_layers=5, num_neurons_per_layer=8):
    '''
    :param num_hidden_layers: numbers of hidden layers
    :param num_neurons_per_layer: numbers of nuerons each layer
    :return: dnn model
    '''
    model = tf.keras.Sequential()  # Initialize a feedforward neural network
    model.add(tf.keras.Input(3))  # Input is two-dimensional

    # Append hidden layers

    model.add(tf.keras.layers.Dense(10,
                                    activation=tf.keras.activations.get('tanh'),
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    for _ in range(num_hidden_layers - 2):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation=tf.keras.activations.get('tanh'),
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(10,
                                    activation=tf.keras.activations.get('tanh'),
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    # Output is two-dimensional
    model.add(tf.keras.layers.Dense(2))

    return model


class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equation.

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
        Computing derivatives for the steady Navier-Stokes equation.

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
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equation.

    Attributes:
        network: keras network model with input (x, y) and output (psi, p).
        rho: density.
        nu: viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, rho=1, nu=0.01):
        """
        Args:
            network: keras network model with input (x, y) and output (psi, p).
            rho: density.
            nu: viscosity.
        """

        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the steady Navier-Stokes equation.

        Returns:
            PINN model for the steady Navier-Stokes equation with
                input: [ (x, y, t) relative to equation,
                         (x, y, t) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero),
                          (psi, psi) relative to boundary condition (psi is duplicated because outputs require the same dimensions),
                          (u, v) relative to boundary condition ]
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
        u_eqn = u * u_x + v * u_y + u_t + p_x / self.rho - self.nu * (u_xx + u_yy)
        v_eqn = u * v_x + v * v_y + v_t + p_y / self.rho - self.nu * (v_xx + v_yy)

        # divergence free condition
        f_div = u_y + v_x

        uv_eqn = tf.concat([u_eqn, v_eqn], axis=-1)
        div_e = tf.concat([f_div, f_div], axis=-1)

        # compute gradients relative to boundary condition
        psi_bnd, _, u_grads_bnd, v_grads_bnd, d_time = self.grads(xyt_bnd)

        # compute boundary condition loss
        psi_bnd = tf.concat([psi_bnd, psi_bnd], axis=-1)
        uv_bnd = tf.concat([u_grads_bnd[0], v_grads_bnd[0]], axis=-1)

        # build PINN model for the time dependent Navier-Stokes equation
        return tf.keras.models.Model(
            inputs=[xyt_eqn, xyt_bnd], outputs=[uv_eqn, div_e, psi_bnd, uv_bnd])


# number of training samples
num_train_samples = 1000
# number of test samples
num_test_samples = 20

# inlet flow velocity
u0 = 1
# density
rho = 1
# viscosity
nu = 0.01

# setting for animation
run_time = 8
number_of_frame = 300

# build a core network model
network = InitializeModel()
network.summary()
# build a PINN model
pinn = PINN(network, rho=rho, nu=nu).build()

# create training input
t = [[run_time * i / num_train_samples] for i in range(num_train_samples)]
xy_eqn = np.random.rand(num_train_samples, 2) * 2 - 1
xyt_eqn = np.hstack((xy_eqn, t))
xy_ub = np.random.rand(num_train_samples // 2, 2)  # top-bottom boundaries
xy_ub[..., 1] = np.round(xy_ub[..., 1])  # y-position is 0 or 1
xy_lr = np.random.rand(num_train_samples // 2, 2)  # left-right boundaries
xy_lr[..., 0] = np.round(xy_lr[..., 0])  # x-position is 0 or 1
xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr])) * 2 - 1
xyt_bnd = np.hstack((xy_bnd, t))
x_train = [xyt_eqn, xyt_bnd]

# create training output
zeros = np.zeros((num_train_samples, 2))
uv_bnd = np.zeros((num_train_samples, 2))
uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])
y_train = [zeros, zeros, zeros, uv_bnd]

# train the model using L-BFGS-B algorithm
lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, maxiter=300)
lbfgs.fit()



def uv(network, xy):
    """
    Compute flow velocities (u, v) for the network with output (psi, p).

    Args:
        xy: network input variables as ndarray.

    Returns:
        (u, v) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u = psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()


def contour(grid, x, y, z, title, levels=50):
    """
    Contour plot.

    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """

    # get the value range
    vmin = np.min(z)
    vmax = np.max(z)
    # plot a contour
    plt.subplot(grid)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)


# create meshgrid coordinates (x, y) for test plots
x = np.linspace(-1, 1, num_test_samples)
y = np.linspace(-1, 1, num_test_samples)

x, y = np.meshgrid(x, y)
data_u = {}
data_psi = {}

for j in range(number_of_frame):
    t = [run_time * j / number_of_frame for i in range(np.square(num_test_samples))]
    xyt = np.stack([x.flatten(), y.flatten(), t], axis=-1)
    # predict (psi, p)
    psi_p = network.predict(xyt, batch_size=len(xyt))
    psi, p = [psi_p[..., i].reshape(x.shape) for i in range(psi_p.shape[-1])]
    # compute (u, v)
    u, v = uv(network, xyt)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    data_u[j] = x, y, u, v
    data_psi[j] = psi.reshape(x.shape)

fig, ax = plt.subplots(1, 2)
# set the distance of two plot
plt.subplots_adjust(wspace=1)


def animate(i):
    # clear present plot on the axis to show animation
    ax[0].clear()
    ax[1].clear()

    _x, _y, _u, _v = data_u[i]
    ax[0].quiver(_x, _y, _u, _v, cmap='plasma')
    ax[1].contourf(x, y, data_psi[i], cmap='plasma')

    # aspect ratio of plot is preserved
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    # set title for subpltos
    ax[0].set_title('Velocity Field')
    ax[1].set_title('Pressure Field')


# Call animate method
ani = animation.FuncAnimation(fig, animate, number_of_frame, interval=50, blit=False)
anis = animation.FFMpegWriter(fps=20)
ani.save('../image/Lid-Driven_.gif', writer='pillow')

# Display the plot
plt.show()
