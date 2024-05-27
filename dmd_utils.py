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

def plot_snapshot(X, Y, Z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis',\
        				edgecolor='green')
    ax.view_init(elev=20, azim=120)
    ax.set_zlim(0 ,0.001)

def dmd(X, Y, truncate=None):
    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r] 
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return mu, Phi

def build_dmd_modes(t, X, Y, Phi, mu, dt=0.001):
    b = dot(pinv(Phi), X[:,0])
    omega = np.diag(np.power(mu, t / dt))
    return dot(Phi, dot(omega ,b))

def SnapshotsToImages(X, Y, snapshots, save_dir="snapshots_dmd_with_1source"):
    os.mkdir(save_dir)
    for index, snapshot in enumerate(snapshots):
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, snapshot, cmap='viridis',\
        				edgecolor='green')
        ax.view_init(elev=20, azim=120)
        ax.set_zlim(0,1)
        plt.savefig(save_dir + '/image' + str(index) + ".png", bbox_inches='tight')
        ax.set_title("max_value: ")

def DMDpredict(phi, mu, time):
    pass

def ModesEvolution(X, T, Phi, mu, dt, x_, y_, save_dir, X_, Y_, gen_3d=False):
    os.mkdir(save_dir)
    b = dot(pinv(Phi), X[:,0])
    modes_count = Phi.shape[1]

    Psi = np.zeros([r, len(T)], dtype="complex")
    for i,_t in enumerate(T):
        Psi[:,i] = multiply(power(mu, _t/dt), b)
    for mode_index in range(modes_count):
        mode = np.reshape(Phi[:, mode_index], (441, 1))
        modes = mode * Psi[mode_index, :]
        snapshots = [np.reshape(modes[:, i], (21, 21)) for i in range(modes.shape[1])]
        SnapshotsToImages(X_, Y_, snapshots, save_dir + "/3d")
        break
        
def matrix_to_snaphots(matrix, save_dir):
    os.mkdir(save_dir)
    snapshots_count = matrix.shape[0]
    for snapshot_index in tqdm(range(snapshots_count), desc="Snapshots generating: "):
        plt.imshow(matrix[snapshot_index, :, :], cmap="plasma")
        plt.savefig(save_dir + "/image" + str(snapshot_index) + ".png")

def save_modes(modes, save_dir):
    os.mkdir(save_dir)
    modes_count = modes.shape[1]
    for i in range(modes_count):
        mode = np.reshape(modes[:, i], (60, 60))
        plt.imshow(mode.real)
        plt.savefig(save_dir + '/image_real' + str(i) + ".png", bbox_inches='tight')
        plt.imshow(mode.imag)
        plt.savefig(save_dir + '/image_imag' + str(i) + ".png", bbox_inches='tight')

def mode_evolution(X, u, T, Phi, r, mu, save_dir, hx, hy, tau, return_evolution_matrix=False):
    T = np.arange(0, 3, 3 / (900))
    T = T[30:]
    os.mkdir(save_dir)
    b = dot(pinv(Phi), X[:,0])
    print("amplitudes:")
    print(b)
    modes_count = Phi.shape[1]
    dt = T[1] - T[0]
    Psi = np.zeros([r, len(T)], dtype="complex")
    for i,_t in enumerate(T):
        Psi[:,i] = multiply(power(mu, _t/dt), b)
    mode_evolution_list = []
    for mode_index in range(modes_count):
        mode = np.reshape(Phi[:, mode_index], (3600, 1))
        modes = mode * Psi[mode_index, :].real
        error = norm(X - modes, ord="fro") / norm(X)
        print("one mode reconstruction error: ", error)
        snapshots = np.array([np.reshape(modes[:, i], (60, 60)) for i in range(modes.shape[1])])
        mode_evolution_list.append(snapshots)
        save_snapshots(snapshots, save_dir + "/mode_" + str(mode_index), format=(60, 60))
    
    plt.scatter(range(len(mode_evolution_list[0][:, 30, 30])), mode_evolution_list[0][:, 30, 30])
    plt.scatter(range(len(mode_evolution_list[0][:, 17, 17])), mode_evolution_list[0][:, 17, 17])
    plt.title("Источник в точке (0.3, 0.3)")
    plt.legend(["Первый (32, 32)", "второй (30, 30)"])
    plt.show()

    if return_evolution_matrix:
        return mode_evolution_list

def matrix_to_surface(matrix, save_dir):
    os.mkdir(save_dir)
    x = np.arange(0, 0.1, 0.1 / 20)
    y = np.arange(0, 0.1, 0.1 / 20)
    X, Y = np.meshgrid(x, y)
    snapshots_count = matrix.shape[0]
    for snapshot_index in tqdm(range(snapshots_count), desc="Surface generating: "):
        plot_snapshot(X, Y, matrix[snapshot_index, :, :])
        plt.savefig(save_dir + "/image" + str(snapshot_index) + ".png")

def calc_freq(matrix, T, point, first=True):
    if first:
        argmax1 = 82
        argmax2 = 271
    else:
        argmax1 = 58
        argmax2 = 182
    
    period = T[argmax2] - T[argmax1]
    if first:
        print("first source")
    else:
        print("second source")
    print("freq: ", 2 * np.pi / period)


def snaphots_research(matrix, r, save_dir, hx, hy, tau, source_info, save_snapshots_=False, save_modes_=False, save_evolution_=False, save_surface_=False, save=True):
    source_matrix = matrix.copy()
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    if save_surface_:
        matrix_to_surface(matrix, save_dir + "/surface")

    if save_snapshots_:
        matrix_to_snaphots(matrix, save_dir + "/snapshots")
        
    modes = np.array([np.reshape(matrix[i, :, :], (3600,)) for i in range(matrix.shape[0])])
    matrix = modes.T
    X, Y = matrix[:, :-1], matrix[:, 1:]
    mu, Phi = dmd(X, Y, truncate=r)
    _, sing_values, _ = svd(matrix)
    print("sing values")
    for i in range(20):
        print(sing_values[i])
    plt.scatter(range(len(sing_values)), sing_values)
    plt.show()
   
    T = np.arange(0, 3, 3 / (900))
    T = T[30:]
    modes, eigs = Phi, mu
    print("aigen values:")
    for value in eigs:
        print(value)
    modes_count = modes.shape[1]
    for i in range(modes_count):
        mode = np.reshape(modes[:, i], (60, 60))
        plt.imshow(mode.real)
        plt.title(str(i))
        plt.show()

    if save_modes_:
        save_modes(modes, save_dir + "/modes")

    if save_evolution_:
        evolution_list = mode_evolution(matrix, source_matrix, T, modes, r, eigs, save_dir + "/modes_evolution", hx, hy, tau, return_evolution_matrix=True)

def save_snapshots(snapshots, save_dir, format=(7, 7)):
    os.mkdir(save_dir)
    m = snapshots.shape[0]
    for index in range(m)[:100]:
        snapshot = snapshots[index]
        snapshot = np.reshape(snapshot, (format[0], format[1]))
        if snapshot.dtype == "complex":
            plt.imshow(snapshot.real, cmap="plasma")
            plt.savefig(save_dir + '/image' + str(index) + "_real" + ".png")
            plt.imshow(snapshot.imag)
            plt.savefig(save_dir + '/image' + str(index) + "_img" + ".png", bbox_inches='tight')
        else:
            plt.imshow(snapshot, cmap="plasma")
            plt.savefig(save_dir + '/image' + str(index)  + ".png")