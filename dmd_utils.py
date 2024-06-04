from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.animation as animation
import os
from operator import mul
from functools import reduce
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv, norm, cond
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn
from itertools import accumulate
from tqdm import tqdm

def makeVideo(path_to_data, movie_name):
    fig = plt.figure()
    sorted_files = sorted(os.listdir(path_to_data))
    files_and_indexes = [(file_name[:-4], int(file_name[:-4].split("image")[1])) for file_name in os.listdir(path_to_data)]
    sorted_files = sorted(files_and_indexes, key=lambda item: item[1])
    image_files = [file[0] for file in sorted_files]
    images = [mpimg.imread(path_to_data + "/" + file_name + ".png") for file_name in image_files]
    frames = []
    for i in range(len(images)):
        frames.append([plt.imshow(images[i])])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat=False)
    
    ani.save(movie_name + ".mp4")

def dmd(X, Y, r):
    U, Sigma, V = svd(X, False)
    U = U[:,:r]
    Sigma = diag(Sigma)[:r,:r] 
    V = V.T[:,:r]
    Atil = dot(dot(dot(U.T, Y), V), inv(Sigma))
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sigma)), W)
    return mu, Phi
        
def matrix_to_snaphots(matrix, save_dir):
    os.mkdir(save_dir)
    snapshots_count = matrix.shape[0]
    for snapshot_index in tqdm(range(snapshots_count), desc="Snapshots generating: "):
        plt.imshow(matrix[snapshot_index, :, :], cmap="plasma")
        plt.savefig(save_dir + "/image" + str(snapshot_index) + ".png")

def save_modes(modes, save_dir, grid_size):
    os.mkdir(save_dir)
    modes_count = modes.shape[1]
    for i in range(modes_count):
        mode = np.reshape(modes[:, i], (grid_size[0], grid_size[1]))
        plt.imshow(mode.real)
        plt.savefig(save_dir + '/image_real' + str(i) + ".png", bbox_inches='tight')
        plt.imshow(mode.imag)
        plt.savefig(save_dir + '/image_imag' + str(i) + ".png", bbox_inches='tight')

def visualize_dynamics(modes, mode_evolution_list, source_info):
    for i, source_location in enumerate(source_info):
        x, y = source_location
        plt.scatter(range(len(mode_evolution_list[modes[i]][:, x, y])), mode_evolution_list[modes[i]][:, x, y])
        plt.title(f"source dynamics in ({x}, {y}) grid point")
        plt.show()

def visualize_modes(modes, grid_size):
    modes_count = modes.shape[1]
    for i in range(modes_count):
        mode = np.reshape(modes[:, i], (grid_size[0], grid_size[1]))
        plt.imshow(mode.real)
        plt.title(str(i))
        plt.show()

def visualize_singular_values(matrix):
    _, sing_values, _ = svd(matrix)
    plt.scatter(range(len(sing_values)), sing_values)
    plt.title("Singular values")
    plt.show()

def mode_evolution(X, grid_size, u, T, Phi, r, mu, save_dir, hx, hy, tau, save_evolution=False):
    dim_size = grid_size[0] * grid_size[1]
    os.mkdir(save_dir)
    b = dot(pinv(Phi), X[:,0])
    modes_count = Phi.shape[1]
    dt = T[1] - T[0]
    Psi = np.zeros([r, len(T)], dtype="complex")
    for i,_t in enumerate(T):
        Psi[:,i] = multiply(power(mu, _t/dt), b)
    mode_evolution_list = []
    for mode_index in range(modes_count):
        mode = np.reshape(Phi[:, mode_index], (dim_size, 1))
        modes = mode * Psi[mode_index, :]
        snapshots = np.array([np.reshape(modes[:, i], (grid_size[0], grid_size[1])) for i in range(modes.shape[1])]).real
        mode_evolution_list.append(snapshots)

        if save_evolution:
            save_snapshots(snapshots, save_dir + "/mode_" + str(mode_index), format=(grid_size[0], grid_size[0]))
    
    source_info = [(17, 17), (50, 50), (33, 33), (27, 27), (30, 30)]
    modes = [11, 9, 4, 2, 6]

    visualize_dynamics(modes, mode_evolution_list, source_info)

    return mode_evolution_list

def relative_error(y_pred, y_true):
    return np.abs(y_pred - y_true) / np.abs(y_true)

def calc_freq(T, t1, t2):
    return 2 * np.pi / (np.abs(T[t1] - T[t2]))

def amplitude_research(evolution_list, source_info, compare_with=None):
    #modes = [10, 8, 4, 2, 6]
    modes = [11, 9, 4, 2, 6]
    a_real = [1.5 * 10e5, 3 * 10e5, 2.5 * 10e5, 1 * 10e6, 2 * 10e5]
    for source_index1, source_location1 in enumerate(source_info):
        for source_index2, source_location2 in enumerate(source_info):
            x1, y1 = source_location1
            x2, y2 = source_location2
            a_pred1 = np.max(evolution_list[modes[source_index1]][:, x1, y1])
            a_pred2 = np.max(evolution_list[modes[source_index2]][:, x2, y2])
            if compare_with is not None:
                a_pred_compare1 = np.max(compare_with[:, x1, y1])
                a_pred_compare2 = np.max(compare_with[:, x2, y2])
            a_true_min = min(a_real[source_index1], a_real[source_index2])
            a_true_max = max(a_real[source_index1], a_real[source_index2])
            relative_true = a_true_max / a_true_min
            relative_pred = max(a_pred1, a_pred2) / min(a_pred1, a_pred2)
            print(f'error sources {source_index1}, {source_index2} from DMD ', relative_error(relative_pred, relative_true), end=" ")
            if compare_with is not None:
                relative_pred = max(a_pred_compare1, a_pred_compare2) / min(a_pred_compare1, a_pred_compare2)
                print("from source data ", relative_error(relative_pred, relative_true))

def snaphots_research(matrix, T, r, save_dir, hx, hy, tau, source_info, save_snapshots_=False, save_modes_=False, save_evolution_=False, save_surface_=False):
    source_matrix = matrix.copy()

    if save_snapshots_ or save_modes_ or save_evolution_:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    if save_snapshots_:
        matrix_to_snaphots(matrix, save_dir + "/snapshots")
    
    grid_size = matrix.shape[1:]
    dim_size = grid_size[0] * grid_size[1]

    for i, source_location in enumerate(source_info):
        x, y = source_location
        plt.scatter(range(len(source_matrix[:, x, y])), source_matrix[:, x, y])
        plt.title(str(i))
        plt.show()

    modes = np.array([np.reshape(matrix[i, :, :], (dim_size,)) for i in range(matrix.shape[0])])
    matrix = modes.T

    X, Y = matrix[:, :-1], matrix[:, 1:]
    mu, Phi = dmd(X, Y, r)
    modes, eigs = Phi, mu

    #visualize_singular_values(matrix)
    
    visualize_modes(modes, grid_size)

    if save_modes_:
        save_modes(modes, save_dir + "/modes", grid_size)

    evolution_list = mode_evolution(matrix, grid_size, source_matrix, T, modes, r, eigs, save_dir + "/modes_evolution", hx, hy, tau, save_evolution=save_evolution_)

    if evolution_list is not None:
        amplitude_research(evolution_list, source_info, compare_with=source_matrix)

    return modes, evolution_list

def save_snapshots(snapshots, save_dir, format=(60, 60), snapshots_count=100):
    os.mkdir(save_dir)
    m = snapshots.shape[0]
    for index in tqdm(range(m)[:snapshots_count], desc="Saving evolution: "):
        snapshot = snapshots[index]
        snapshot = np.reshape(snapshot, (format[0], format[1]))
        if snapshot.dtype == "complex":
            plt.imshow(snapshot.real, cmap="plasma")
            plt.savefig(save_dir + '/image' + str(index) + "_real" + ".png")
            #plt.imshow(snapshot.imag)
            #plt.savefig(save_dir + '/image' + str(index) + "_img" + ".png", bbox_inches='tight')
        else:
            plt.imshow(snapshot, cmap="plasma")
            plt.savefig(save_dir + '/image' + str(index)  + ".png")