import numpy as np
import matplotlib.pyplot as plt
from dmd_utils import matrix_to_snaphots
from dmd_utils import snaphots_research
from dmd_utils import dmd
from numpy.linalg import cond
from tqdm import tqdm
import copy
from sklearn.preprocessing import StandardScaler

class Task:
    def __init__(self, **params):
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.f = params['f']
        
        self.alpha_x0 = params['alpha_x0']
        self.beta_x0 = params['beta_x0']
        self.f_phi_x0 = params['f_phi_x0']
        
        self.alpha_xl = params['alpha_xl']
        self.beta_xl = params['beta_xl']
        self.f_phi_xl = params['f_phi_xl']
        
        self.alpha_y0 = params['alpha_y0']
        self.beta_y0 = params['beta_y0']
        self.f_phi_y0 = params['f_phi_y0']
        
        self.alpha_yl = params['alpha_yl']
        self.beta_yl = params['beta_yl']
        self.f_phi_yl = params['f_phi_yl']
        
        self.f_psi = params['f_psi']

        self.interval_x = params['interval_x']
        self.interval_y = params['interval_y']
        self.analytical_solution = params['analytical_solution']
        
class Solver:
    
    @classmethod
    def solve(cls, task:Task, nx, ny, nt, max_time, method):
        hx, hy, tau, time_cnt = cls.__prepare(task.interval_x[0], task.interval_x[1], nx,
                                              task.interval_y[0], task.interval_y[1], ny,
                                              task.a, task.b, max_time)
        tau = max_time / nt
        time_cnt = nt
        if method == 'alternative_directions':
            u = cls.__alternative_directions_method(task, nx, ny, time_cnt, hx, hy, tau)
        elif method == 'fractional_steps':
            u = cls.__fractional_steps_method(task, nx, ny, time_cnt, hx, hy, tau)
        x, y = cls.__get_split_by_space(hx, nx, hy, ny)
        t = cls.__get_split_by_time(tau, time_cnt)
        return u, x, y, t, hx, hy, tau
    
    @classmethod
    def __alternative_directions_method(cls, task:Task, nx, ny, time_cnt, hx, hy, tau):
        u = np.zeros((time_cnt, nx, ny))
        
        a_1 = np.array([0.0] + [-task.a * tau * hy ** 2 for _ in range(1, nx - 1)] + [0.0])
        b_1 = np.array([0.0] + [2 * hx ** 2 * hy ** 2 * task.c + 2 * task.a * hy ** 2 * tau for _ in range(1, nx - 1)] + [0.0])
        c_1 = np.array([0.0] + [-task.a * tau * hy ** 2 for _ in range(1, nx - 1)] + [0.0])
        
        b_1[0] = hx * task.beta_x0 - task.alpha_x0
        c_1[0] = task.alpha_x0
        a_1[-1] = -task.alpha_xl
        b_1[-1] = task.alpha_xl + hx * task.beta_xl
        
        a_2 = np.array([0.0] + [-task.b * tau * hx ** 2 for _ in range(1, ny - 1)] + [0.0])
        b_2 = np.array([0.0] + [2 * hx ** 2 * hy ** 2 * task.c + 2 * task.b * hx ** 2 * tau for _ in range(1, ny - 1)] + [0.0])
        c_2 = np.array([0.0] + [-task.b * tau * hx ** 2 for _ in range(1, ny - 1)] + [0.0])
        
        b_2[0] = hy * task.beta_y0 - task.alpha_y0
        c_2[0] = task.alpha_y0
        a_2[-1] = -task.alpha_yl
        b_2[-1] = task.alpha_yl + hy * task.beta_yl
        
        for i in range(nx):
            for j in range(ny):
                u[0][i][j] = task.f_psi(hx * i, hy * j)
                
        for k in tqdm(range(1, time_cnt), desc="Time progress: "):
            u_1 = np.zeros((ny, nx))
            d_1 = np.zeros(nx)
            u_2 = np.zeros((nx, ny))
            d_2 = np.zeros(ny)
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    d_1[i] = (u[k - 1][i][j - 1] - 2 * u[k - 1][i][j] + u[k - 1][i][j + 1]) * task.b * hx ** 2 * tau + \
                              2 * hx ** 2 * hy ** 2 * task.c * u[k - 1][i][j] + \
                              hx ** 2 * hy ** 2 * tau * task.f(i, j, tau * (k - 0.5))
                d_1[0] = hx * task.f_phi_x0(hy * j ,tau * (k - 0.5))
                d_1[-1] = hx * task.f_phi_xl(hy * j ,tau * (k - 0.5))
                
                u_1_j = cls.__tma(a_1, b_1, c_1, d_1, nx)
                u_1[j] = copy.deepcopy(u_1_j)
            for i in range(nx):
                u_1[0][i] = (hy * task.f_phi_y0(i * hx, tau * (k - 0.5)) - task.alpha_y0 * u_1[1][i]) / (hy * task.beta_y0 - task.alpha_y0)
                u_1[-1][i] = (hy * task.f_phi_yl(i * hx, tau * (k - 0.5)) + task.alpha_yl * u_1[-2][i]) / (hy * task.beta_yl + task.alpha_yl)
            u_1 = u_1.transpose()
            
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    d_2[j] = (u_1[i - 1][j] - 2 * u_1[i][j] + u_1[i + 1][j]) * task.a * hy ** 2 * tau + \
                              2 * hx ** 2 * hy ** 2 * task.c * u_1[i][j] + \
                              hx ** 2 * hy ** 2 * tau * task.f(i, j, tau * k)
                d_2[0] = hy * task.f_phi_y0(hx * i ,tau * k)
                d_2[-1] = hy * task.f_phi_yl(hx * i ,tau * k)
                
                u_2_i = cls.__tma(a_2, b_2, c_2, d_2, ny)
                u_2[i] = copy.deepcopy(u_2_i)
            for j in range(ny):
                u_2[0][j] = (hx * task.f_phi_x0(j * hy, tau * k) - task.alpha_x0 * u_2[1][j]) / (hx * task.beta_x0 - task.alpha_x0)
                u_2[-1][j] = (hx * task.f_phi_xl(j * hy, tau * k) + task.alpha_xl * u_2[-2][j]) / (hx * task.beta_xl + task.alpha_xl)
            u[k] = copy.deepcopy(u_2)
        return u
    
    @classmethod
    def __fractional_steps_method(cls, task:Task, nx, ny, time_cnt, hx, hy, tau):
        u = np.zeros((time_cnt, nx, ny))
        
        a_1 = np.array([0.0] + [-2 * task.a * tau for _ in range(1, nx - 1)] + [0.0])
        b_1 = np.array([0.0] + [2 * hx ** 2 + 4 * tau * task.a for _ in range(1, nx - 1)] + [0.0])
        c_1 = np.array([0.0] + [-2 * task.a * tau for _ in range(1, nx - 1)] + [0.0])
        
        b_1[0] = hx * task.beta_x0 - task.alpha_x0
        c_1[0] = task.alpha_x0
        a_1[-1] = -task.alpha_xl
        b_1[-1] = task.alpha_xl + hx * task.beta_xl
        
        a_2 = np.array([0.0] + [-2 * task.b * tau for _ in range(1, ny - 1)] + [0.0])
        b_2 = np.array([0.0] + [2 * hy ** 2 + 4 * tau * task.b for _ in range(1, ny - 1)] + [0.0])
        c_2 = np.array([0.0] + [-2 * task.b * tau for _ in range(1, ny - 1)] + [0.0])
        
        b_2[0] = hy * task.beta_y0 - task.alpha_y0
        c_2[0] = task.alpha_y0
        a_2[-1] = -task.alpha_yl
        b_2[-1] = task.alpha_yl + hy * task.beta_yl
        
        for i in range(nx):
            for j in range(ny):
                u[0][i][j] = task.f_psi(hx * i, hy * j)
                
        for k in range(1, time_cnt):
            u_1 = np.zeros((ny, nx))
            d_1 = np.zeros(nx)
            u_2 = np.zeros((nx, ny))
            d_2 = np.zeros(ny)
            for j in range (1, ny - 1):
                for i in range(1, nx - 1):
                    d_1[i] = 2 * hx ** 2 * u[k - 1][i][j] + tau * hx ** 2 * task.f(hx * i, hy * j, tau * (k - 0.5))
                d_1[0] = hx * task.f_phi_x0(hy * j ,tau * (k - 0.5))
                d_1[-1] = hx * task.f_phi_xl(hy * j ,tau * (k - 0.5))
                
                u_1_j = cls.__tma(a_1, b_1, c_1, d_1, nx)
                u_1[j] = copy.deepcopy(u_1_j)
            for i in range(nx):
                u_1[0][i] = (hy * task.f_phi_y0(i * hx, tau * (k - 0.5)) - task.alpha_y0 * u_1[1][i]) / (hy * task.beta_y0 - task.alpha_y0)
                u_1[-1][i] = (hy * task.f_phi_yl(i * hx, tau * (k - 0.5)) + task.alpha_yl * u_1[-2][i]) / (hy * task.beta_yl + task.alpha_yl)
            u_1 = u_1.transpose()
            
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    d_2[j] = 2 * hy ** 2 * u_1[i][j] + tau * hy ** 2 * task.f(hx * i, hy * j, tau * k)
                d_2[0] = hy * task.f_phi_y0(hx * i ,tau * k)
                d_2[-1] = hy * task.f_phi_yl(hx * i ,tau * k)
                
                u_2_i = cls.__tma(a_2, b_2, c_2, d_2, ny)
                u_2[i] = copy.deepcopy(u_2_i)
            for j in range(ny):
                u_2[0][j] = (hx * task.f_phi_x0(j * hy, tau * k) - task.alpha_x0 * u_2[1][j]) / (hx * task.beta_x0 - task.alpha_x0)
                u_2[-1][j] = (hx * task.f_phi_xl(j * hy, tau * k) + task.alpha_xl * u_2[-2][j]) / (hx * task.beta_xl + task.alpha_xl)
            u[k] = copy.deepcopy(u_2)
        return u
                
    @classmethod
    def analytical_solution(cls, task:Task, nx, ny, max_time):
        hx, hy, tau, time_cnt = cls.__prepare(task.interval_x[0], task.interval_x[1], nx,
                                              task.interval_y[0], task.interval_y[1], ny,
                                              task.a, task.b, max_time)
        res = np.zeros((time_cnt, nx, ny))
        for k in range(time_cnt):
            for i in range(nx):
                for j in range(ny):
                    res[k][i][j] = task.analytical_solution(hx * i, hy * j, tau * k)
        return res
            
    @classmethod
    def __tma(cls, a, b, c, d, n):
        A = np.zeros(n)
        B = np.zeros(n)
        x = np.zeros(n)
        A[0] = -c[0] / b[0]
        B[0] = d[0] / b[0]
        
        for j in range(1, n):
            A[j] = -c[j] / (b[j] + a[j] * A[j - 1])
            B[j] = (d[j] - a[j] * B[j - 1]) / (b[j] + a[j] * A[j - 1])
        x[-1] = B[-1]
        for j in range(n - 2, -1, -1):
            x[j] = A[j] * x[j + 1] + B[j]
        return x
    
    @classmethod
    def __get_split_by_space(cls, hx, nx, hy, ny):
        return np.array([hx * i for i in range(nx)]), np.array([hy * j for j in range(ny)])
    
    @classmethod
    def __get_split_by_time(cls, tau, time_cnt):
        return np.array([tau * k for k in range(time_cnt)])
    
    @classmethod
    def __prepare(cls, begin_x, end_x, nx, begin_y, end_y, ny, a, b, max_time):
        hx = (end_x - begin_x) / (nx - 1)
        hy = (end_y - begin_y) / (ny - 1)
        tau = cls.__sigma / (a / hx ** 2 + b / hy ** 2)
        time_cnt = int(max_time / tau)
        return hx, hy, tau, time_cnt
    
    __sigma = 0.4

if __name__ == "__main__":
    '''
    params1 = {
        'a': 10.0,
        'b': 10.0,
        'c': 1000.0,
        'f': lambda i, j, t: 100 * np.sin(15 * (np.pi) * t + np.pi / 2) if i == 20 and j == 20 else 0.0,

        'alpha_x0': 0.0,
        'beta_x0': 1.0,
        'f_phi_x0': lambda y, t: 0.0,

        'alpha_xl': 0.0,
        'beta_xl': 1.0,
        'f_phi_xl': lambda y, t: 0.0,

        'alpha_y0': 0.0,
        'beta_y0': 1.0,
        'f_phi_y0': lambda x, t: 0.0,

        'alpha_yl': 0.0,
        'beta_yl': 1.0,
        'f_phi_yl': lambda x, t: 0.0,

        'f_psi': lambda x, y: 0.0,

        'interval_x': (0.0, 0.3),
        'interval_y': (0.0, 0.3),
        'analytical_solution': lambda x, y, t: np.cos(x) * np.cos(y) * np.exp(-2 * t)
    }
    '''

    params1 = {
    'a': 1.0,
    'b': 1.0,
    'c': 1.0,
    'f': lambda x, y, t: 0,

    'alpha_x0': 0.0,
    'beta_x0': 1.0,
    'f_phi_x0': lambda y, t: np.cos(y) * np.exp(-2 * t),

    'alpha_xl': 0.0,
    'beta_xl': 1.0,
    'f_phi_xl': lambda y, t: -np.cos(y) * np.exp(-2 * t),

    'alpha_y0': 0.0,
    'beta_y0': 1.0,
    'f_phi_y0': lambda x, t: np.cos(x) * np.exp(-2 * t),

    'alpha_yl': 0.0,
    'beta_yl': 1.0,
    'f_phi_yl': lambda x, t: -np.cos(x) * np.exp(-2 * t),

    'f_psi': lambda x, y: np.cos(x) * np.cos(y),

    'interval_x': (0.0, np.pi),
    'interval_y': (0.0, np.pi),
    'analytical_solution': lambda x, y, t: np.cos(x) * np.cos(y) * np.exp(-2 * t)
    }


    # main function to test: f(t) = 1000 * np.cos(10 * t + np.pi / 2)

    params7 = {
    'a': 10.0,
    'b': 10.0,
    'c': 1000.0,
    'f': lambda x, y, t: 1000 * np.cos(10 * t + np.pi / 2) if x == 10 and y == 10 else 0.0,

    'alpha_x0': 0.0,
    'beta_x0': 1.0,
    'f_phi_x0': lambda y, t: 0,

    'alpha_xl': 0.0,
    'beta_xl': 1.0,
    'f_phi_xl': lambda y, t: 0,

    'alpha_y0': 0.0,
    'beta_y0': 1.0,
    'f_phi_y0': lambda x, t: 0,

    'alpha_yl': 0.0,
    'beta_yl': 1.0,
    'f_phi_yl': lambda x, t: 0,

    'f_psi': lambda x, y: 0.001,#1000000 if x == 10 and y == 10 else 0.000,

    'interval_x': (0.0, 0.1),
    'interval_y': (0.0, 0.1),
    'analytical_solution': lambda x, y, t: x * y * np.cos(t)
}

def function_field(x, y, t):
    if x == 5 and y == 5:
        return 150000 * np.cos(10 * t + np.pi / 3)
    elif x == 15 and y == 15:
        return 300000 * np.cos(20 * t + np.pi / 8)
    elif x == 13 and y == 13:
        return 10000 * np.cos(0.1 * t + np.pi / 4)
    elif x == 11 and y == 12:
        return 1000000 * np.cos(150 * t + np.pi / 2)
    elif x == 10 and y == 10:
        return 200000 * np.cos(40 * t + np.pi / 6)
    else:
        return 0
    
#def function_field(x, y, t):
#    if x == 30 and y == 30:
#        return 100000 * np.cos(20 * t)
#    elif x == 32 and y == 32:
#        return 400000 * np.cos(30 * t)
#    else:
#        return 0
    
def function_field(x, y, t):
    if x == 17 and y == 17:
        return 150000 * np.cos(5 * t + np.pi / 3)
    elif x == 50 and y == 50:
        return 300000 * np.cos(10 * t + np.pi / 8)
    elif x == 33 and y == 33:
        return 250000 * np.cos(40 * t + np.pi / 4)
    elif x == 27 and y == 27:
        return 1000000 * np.cos(60 * t + np.pi / 2)
    elif x == 30 and y == 30:
        return 200000 * np.cos(20 * t + np.pi / 6)
    else:
        return 0
    
def function_field(x, y, t):
    if x == 17 and y == 17:
        return 150000 * np.cos(5 * t + np.pi / 3) # 10
    elif x == 50 and y == 50:
        return 300000 * np.cos(10 * t + np.pi / 8) # 20
    elif x == 33 and y == 33:
        return 250000 * np.cos(40 * t + np.pi / 4) # 80
    elif x == 27 and y == 27:
        return 1000000 * np.cos(60 * t + np.pi / 2) # 150
    elif x == 30 and y == 30:
        return 200000 * np.cos(20 * t + np.pi / 6) # 40
    else:
        return 0


params8 = {
    'a': 10.0,
    'b': 10.0,
    'c': 1000.0,
    'f': lambda x, y, t: function_field(x, y, t),

    'alpha_x0': 0.0,
    'beta_x0': 1.0,
    'f_phi_x0': lambda y, t: 0,

    'alpha_xl': 0.0,
    'beta_xl': 1.0,
    'f_phi_xl': lambda y, t: 0,

    'alpha_y0': 0.0,
    'beta_y0': 1.0,
    'f_phi_y0': lambda x, t: 0,

    'alpha_yl': 0.0,
    'beta_yl': 1.0,
    'f_phi_yl': lambda x, t: 0,

    'f_psi': lambda x, y: 0.001,#1000000 if x == 10 and y == 10 else 0.000,

    'interval_x': (0.0, 0.1),
    'interval_y': (0.0, 0.1),
    'analytical_solution': lambda x, y, t: x * y * np.cos(t)
}

def function_field1(x, y, t):
    if x == 7 and y == 7:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 13 and y == 13:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 15 and y == 5:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 5 and y == 15:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 10 and y == 10:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 5 and y == 5:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 10 and y == 15:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 15 and y == 10:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 4 and y == 10:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 10 and y == 5:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 7 and y == 10:
        return 10000 * np.cos(20 * t + np.pi / 2)
    elif x == 17 and y == 15:
        return 10000 * np.cos(20 * t + np.pi / 2)
    else:
        return 0

params9 = {
    'a': 10.0,
    'b': 10.0,
    'c': 1000.0,
    'f': lambda x, y, t: function_field1(x, y, t),

    'alpha_x0': 0.0,
    'beta_x0': 1.0,
    'f_phi_x0': lambda y, t: 0,

    'alpha_xl': 0.0,
    'beta_xl': 1.0,
    'f_phi_xl': lambda y, t: 0,

    'alpha_y0': 0.0,
    'beta_y0': 1.0,
    'f_phi_y0': lambda x, t: 0,

    'alpha_yl': 0.0,
    'beta_yl': 1.0,
    'f_phi_yl': lambda x, t: 0,

    'f_psi': lambda x, y: 0.001,#1000000 if x == 10 and y == 10 else 0.000,

    'interval_x': (0.0, 0.1),
    'interval_y': (0.0, 0.1),
    'analytical_solution': lambda x, y, t: x * y * np.cos(t)
}

def function_field2(x, y, t):
    if x == 30 and y == 30:
        return 100000 * np.cos(20 * t)
    elif x == 32 and y == 32:
        return 400000 * np.cos(32 * t)
    else:
        return 0

params10 = {
    'a': 10.0,
    'b': 10.0,
    'c': 1000.0,
    'f': lambda x, y, t: function_field2(x, y, t),

    'alpha_x0': 0.0,
    'beta_x0': 1.0,
    'f_phi_x0': lambda y, t: 0,

    'alpha_xl': 0.0,
    'beta_xl': 1.0,
    'f_phi_xl': lambda y, t: 0,

    'alpha_y0': 0.0,
    'beta_y0': 1.0,
    'f_phi_y0': lambda x, t: 0,

    'alpha_yl': 0.0,
    'beta_yl': 1.0,
    'f_phi_yl': lambda x, t: 0,

    'f_psi': lambda x, y: 0.001,#1000000 if x == 10 and y == 10 else 0.000,

    'interval_x': (0.0, 0.1),
    'interval_y': (0.0, 0.1),
    'analytical_solution': lambda x, y, t: x * y * np.cos(t)
}

task = Task(**params8)
method = "alternative_directions"
nx = 60
ny = 60
nt = 900
max_time = 3

u,x_p,y_p,times, hx, hy, tau= Solver.solve(task=task, nx=nx, ny=ny, nt=nt, max_time=max_time, method=method)#max_time=max_time, method=method)

source_info = [(17, 17), (50, 50), (33, 33), (27, 27), (30, 30)]

T = np.arange(0, max_time, max_time / (nt))

modes = np.array([np.reshape(u[i, :, :], (3600,)) for i in range(u.shape[0])])
matrix = modes.T
#dmd(matrix[:, :-1], matrix[:, 1:], truncate=3)
snaphots_research(u[30:, :, :], T[30:], 13, "check_evol_working", hx, hy, tau, source_info, save_modes_=False, save_snapshots_=False, save_evolution_=True)