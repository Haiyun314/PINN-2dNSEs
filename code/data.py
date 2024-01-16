import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


class Data:
    """
    use test_data method to view the points' distribution.
    """
    @staticmethod
    def lid_driven_cavity():
        # create training input
        t = [[RUN_TIME * i / NUM_TRAIN_SAMPLES] for i in range(NUM_TRAIN_SAMPLES)]
        xy_ub = np.random.rand(NUM_TRAIN_SAMPLES // 2, 2)  # top-bottom boundaries
        xy_ub[..., 1] = np.round(xy_ub[..., 1])  # y-position is 0 or 1
        xy_lr = np.random.rand(NUM_TRAIN_SAMPLES // 2, 2)  # left-right boundaries
        xy_lr[..., 0] = np.round(xy_lr[..., 0])  # x-position is 0 or 1
        xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr])) * 2 - 1
        boundary_points = np.hstack((xy_bnd, t))
        x_train = [boundary_points, boundary_points]

        # create training output
        zeros = np.zeros((NUM_TRAIN_SAMPLES, 2))
        uv_bnd = np.zeros((NUM_TRAIN_SAMPLES, 2))
        uv_bnd[..., 0] = U0 * np.floor(xy_bnd[..., 1])
        y_train = [zeros, zeros, zeros, uv_bnd]
        print('Data prepared\n')
        return x_train, y_train

    @staticmethod
    def pipe():
        init_data = np.empty((0, 3))
        x = np.linspace(-1, 1, NUM_TEST_SAMPLES)
        y = np.linspace(-1, 1, NUM_TEST_SAMPLES)
        x, y = np.meshgrid(x, y)
        for j in range(NUMBER_OF_FRAMES):
            t = [RUN_TIME * j / NUMBER_OF_FRAMES for _ in range(np.square(NUM_TEST_SAMPLES))]
            xyt = np.stack([x.flatten(), y.flatten(), t], axis=-1)
            init_data = np.concatenate((init_data, xyt))
        mask_condition = ((0, 1), (0, -1), (1, 1), (1, -1))
        # masks at x=1, x=-1, y= 1, y= -1
        masks = [[init_data[:, col] == val] for col, val in mask_condition]
        mask = np.logical_or.reduce(masks)[0]
        boundary_points = init_data[mask]
        boundary_points = np.tile(boundary_points, (5, 1))[:16200]
        np.random.shuffle(boundary_points)
        interior_points = init_data[~mask]
        x_train = [interior_points, boundary_points]
        zeros_inter = np.zeros((len(interior_points), 2))  # interior NSEs and div free conditions
        zeros_bnd = np.zeros((len(boundary_points), 2))  # boundary velocity conditions
        # points at the line x = 1 and x = -1
        mask_bnd = np.logical_or(boundary_points[:, 0] == 1, boundary_points[:, 0] == -1)
        # set the points at x1 and x2 with y directional velocity equal to 0, x directional velocity equal to 1
        zeros_bnd_psi = np.copy(zeros_bnd)
        zeros_bnd[mask_bnd, 1] = 1
        y_train = [zeros_inter, zeros_inter, zeros_bnd_psi, zeros_bnd]
        print('Data prepared\n')
        return x_train, y_train

    @staticmethod
    def test_model():
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
            data_u[j] = u, v
            data_psi[j] = psi.reshape(x.shape)
        return data_u, data_psi, (x, y)

    @staticmethod
    def test_data(interior_points, boundary_points):
        plt.scatter(interior_points[:, 0], interior_points[:, 1], c='red')
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='green')
        plt.show()

