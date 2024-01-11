import numpy as np
import tensorflow as tf

# number of training samples
NUM_TRAIN_SAMPLES = 1000
# number of test samples
NUM_TEST_SAMPLES = 20

MAX_ITER = 100
# inlet flow velocity
U0 = 1
# density
RHO = 1
# viscosity
NU = 0.1

# setting for animation
RUN_TIME = 4
NUMBER_OF_FRAMES = 50


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


class TrainData:
    @staticmethod
    def lid_driven_cavity():
        # create training input
        t = [[RUN_TIME * i / NUM_TRAIN_SAMPLES] for i in range(NUM_TRAIN_SAMPLES)]
        xy_eqn = np.random.rand(NUM_TRAIN_SAMPLES, 2) * 2 - 1
        xyt_eqn = np.hstack((xy_eqn, t))
        xy_ub = np.random.rand(NUM_TRAIN_SAMPLES // 2, 2)  # top-bottom boundaries
        xy_ub[..., 1] = np.round(xy_ub[..., 1])  # y-position is 0 or 1
        xy_lr = np.random.rand(NUM_TRAIN_SAMPLES // 2, 2)  # left-right boundaries
        xy_lr[..., 0] = np.round(xy_lr[..., 0])  # x-position is 0 or 1
        xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr])) * 2 - 1
        xyt_bnd = np.hstack((xy_bnd, t))
        x_train = [xyt_eqn, xyt_bnd]

        # create training output
        zeros = np.zeros((NUM_TRAIN_SAMPLES, 2))
        uv_bnd = np.zeros((NUM_TRAIN_SAMPLES, 2))
        uv_bnd[..., 0] = U0 * np.floor(xy_bnd[..., 1])
        y_train = [zeros, zeros, zeros, uv_bnd]
        return x_train, y_train


class TestData:
    @staticmethod
    def test_data():
        try:
            network = tf.keras.models.load_model('./pinn')
        except FileNotFoundError:
            raise FileNotFoundError("can't find the pinn model")
        # create meshgrid coordinates (x, y) for test plots
        x = np.linspace(-1, 1, NUM_TEST_SAMPLES)
        y = np.linspace(-1, 1, NUM_TEST_SAMPLES)

        x, y = np.meshgrid(x, y)
        data_u = {}
        data_psi = {}

        for j in range(NUMBER_OF_FRAMES):
            t = [RUN_TIME * j / NUMBER_OF_FRAMES for i in range(np.square(NUM_TEST_SAMPLES))]
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
        return data_u, data_psi
