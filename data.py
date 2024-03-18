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
    Utilize the "test_data" method to inspect the dataset.
    """

    @staticmethod
    def lid_driven_cavity():
        """
        To generate the training graph, where one side has directional velocity one, all
        other components equal to zero.
        :return: The training positions and target outputs.
        """
        # create training input
        t = [[RUN_TIME * i / NUM_TRAIN_SAMPLES] for i in range(NUM_TRAIN_SAMPLES)]
        xy_eqn = np.random.rand(NUM_TRAIN_SAMPLES, 2) * 2 -1
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
        mask = xy_bnd[..., 0] == -1
        uv_bnd[mask, 1] = 1
        y_train = [zeros, zeros, zeros, uv_bnd]
        return x_train, y_train, Data.lid_driven_cavity.__name__

    @staticmethod
    def pipe():
        """
        To generate the Pipe graph, where the inlet and outlet have same directional velocity one, all other
        components equal to zero.
        :return: The training positions and target outputs.
        """
        init_data = np.empty((0, 3))
        x = np.linspace(-1, 1, NUM_TEST_SAMPLES)
        y = np.linspace(-0.5, 0.5, NUM_TEST_SAMPLES)
        x, y = np.meshgrid(x, y)
        for j in range(NUMBER_OF_FRAMES):
            t = [RUN_TIME * j / NUMBER_OF_FRAMES for _ in range(np.square(NUM_TEST_SAMPLES))]
            xyt = np.stack([x.flatten(), y.flatten(), t], axis=-1)
            init_data = np.concatenate((init_data, xyt))

        mask_bound_condition = ((1, 0.5), (1, -0.5))
        # masks on y= 1, y= -1
        masks = [[init_data[:, col] == val] for col, val in mask_bound_condition]
        mask = np.logical_or.reduce(masks)[0]
        interior_points = init_data[~mask]
        boundary_points = init_data[mask]
        boundary_points = np.tile(boundary_points, (len(interior_points)//len(boundary_points), 1))[:len(interior_points)]  # to get same the number of points
        np.random.shuffle(boundary_points)
        x_train = [interior_points, boundary_points]
        zeros_interior = np.zeros((len(interior_points), 2))  # interior NSEs and div free conditions
        zeros_bnd = np.zeros((len(boundary_points), 2))  # boundary velocity conditions
        # points on the line x = -1
        mask_inlet = interior_points[:, 0] == -1
        zeros_bnd_psi = np.copy(zeros_bnd)
        zeros_interior_div = np.copy(zeros_interior) # divergence free condition
        zeros_interior[mask_inlet, 0] = 1 # set inlet x directional NSE equal to 1
        y_train = [zeros_interior, zeros_interior_div, zeros_bnd_psi, zeros_bnd]
        print('Data prepared\n')
        return x_train, y_train, Data.pipe.__name__

    @staticmethod
    def test_model(name: str):
        try:
            network = tf.keras.models.load_model(f'./{name}')
        except FileNotFoundError:
            raise FileNotFoundError("can't find the pinn model")
        # create meshgrid coordinates (x, y) for test plots
        x = np.linspace(-1, 1, NUM_TEST_SAMPLES)
        y = np.linspace(-0.5, 0.5, NUM_TEST_SAMPLES)

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
        return data_u, data_psi, (x, y), name

    @staticmethod
    def test_data(data):
        input_data, output_data = data
        fig, ax = plt.subplots(1, 2)
        interior_points, boundary_points = input_data
        # pick up the conditions for interior equations and boundary velocity.
        interior_outputs = output_data[0]
        boundary_outputs = output_data[3]

        ax[0].scatter(interior_points[:, 0], interior_points[:, 1], c='red', alpha=0.6)
        ax[0].scatter(boundary_points[:, 0], boundary_points[:, 1], c='green', alpha=0.2)
        ax[0].set_title('input_data')
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[1].quiver(interior_points[:, 0], interior_points[:, 1], interior_outputs[:, 0],
                     interior_outputs[:, 1], scale=3, color='red', alpha=0.6)
        ax[1].quiver(boundary_points[:, 0], boundary_points[:, 1], boundary_outputs[:, 0],
                     boundary_outputs[:, 1], scale=3, color='green', alpha=0.6)
        ax[1].set_title('output_data')

        plt.show()


# data = Data.pipe()
# Data.test_data(data[:2])

