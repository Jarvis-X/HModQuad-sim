from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import time
from itertools import combinations
import math
import numpy as np
import scipy as sp


class GrowingStructure:
    def __init__(self, max_num_modules=5, configuration=np.zeros((5, 5)), module_config=None):
        # the binary matrix that marks the position of existing modules
        self._max_n = max_num_modules
        self._configuration = configuration
        self._possible_new_structure = set()
        if not np.sum(self._configuration) == 0:
            self._possible_new_structure.add(totuple(configuration))
        self._n = np.sum(self._configuration)

    def get_num_modules(self):
        return self._n

    def grow(self, heuristic=True):
        center = (self._max_n - 1)/2
        # print(center)
        if len(self._possible_new_structure) == 0:
            # print("!!")
            # print(self._possible_new_structure)
            if not self._max_n % 2 == 0:
                # odd maximum number of modules -> 1x1 initial configuration
                configuration = np.zeros((self._max_n, self._max_n))
                configuration[int(center), int(center)] = 1
            else:
                # even maximum number of modules -> 2x2 initial configuration for heuristic
                if heuristic:
                    configuration = np.zeros((self._max_n, self._max_n))
                    configuration[int(np.floor(center)):int(np.ceil(center)+1), int(np.floor(center)):int(np.ceil(center)+1)] = 1
                else:
                    configuration = np.zeros((self._max_n, self._max_n))
                    configuration[int(np.floor(center)), int(np.floor(center))] = 1
            self._possible_new_structure.add(totuple(configuration))
        else:
            new_possible_new_structure = set()
            new_possible_new_structure_relative = set()  # to avoid repetitions in non-heuristic version
            for structure_tuple in self._possible_new_structure:
                structure = np.array(structure_tuple)
                if heuristic:
                    if np.sum(structure) >= self._max_n-1:
                        print(structure)
                        return
                else:
                    if np.sum(structure) >= self._max_n:
                        return
                surfaces = set()
                for i in range(structure.shape[0]):
                    for j in range(structure.shape[1]):
                        if structure[i, j] == 1:
                            if i + 1 < self._max_n:
                                if structure[i + 1, j] == 0:
                                    surfaces.add((i + 1, j))
                            if j + 1 < self._max_n:
                                if structure[i, j + 1] == 0:
                                    surfaces.add((i, j + 1))
                            if i - 1 >= 0:
                                if structure[i - 1, j] == 0:
                                    surfaces.add((i - 1, j))
                            if j - 1 >= 0:
                                if structure[i, j - 1] == 0:
                                    surfaces.add((i, j - 1))

                for surface in surfaces:
                    new_structure = structure
                    # heuristic: symmetry
                    if heuristic:
                        # We do not need to consider repetition for symmetric structures since their CoM does not shift
                        new_structure[surface] = 1
                        new_structure[int(2*center-surface[0]), int(2*center-surface[1])] = 1
                        new_possible_new_structure.add(totuple(new_structure))
                        new_structure[surface] = 0
                        new_structure[int(2 * center - surface[0]), int(2 * center - surface[1])] = 0
                    else:
                        new_structure[surface] = 1
                        abstraction = new_structure[~np.all(new_structure == 0, axis=1)]
                        abstraction = abstraction[:, ~np.all(abstraction == 0, axis=0)]
                        if totuple(abstraction) in new_possible_new_structure_relative:
                            new_structure[surface] = 0
                        else:
                            new_possible_new_structure.add(totuple(new_structure))
                            new_possible_new_structure_relative.add(totuple(abstraction))
                            new_structure[surface] = 0
            self._possible_new_structure = new_possible_new_structure
            if heuristic:
                # print("??")
                self._n += 2
            else:
                self._n += 1

    def get_config(self):
        return [np.array(structure) for structure in self._possible_new_structure]

def totuple(input_list):
    return tuple(tuple(input_list[i]) for i in range(len(input_list)))


def RotY(th):
    R = np.array([[np.cos(th), 0, np.sin(th)],
                  [0, 1, 0],
                  [np.sin(th), 0, np.cos(th)]])
    return R


def RotX(th):
    R = np.array([[1, 0, 0],
                [0, np.cos(th), -np.sin(th)],
                [0, np.sin(th), np.cos(th)]])
    return R


def config_to_pose(config, n, l, verbose):
    max_n = config.shape[0]
    com_index = [0, 0]
    for j in range(2):
        collapsed_config = np.sum(config, j)
        index_sum = 0
        for i in range(max_n):
            # print(collapsed_config[i])
            index_sum += i * collapsed_config[i]
        com_index[1 - j] = index_sum / n
    # com_index gives the center of mass of the configuration in the nxn grid
    # [x, y] or [col, row]
    # print(com_index)

    # then we obtain p_{ij}
    p = np.zeros([3, int(4 * n)])
    count = 0
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            if config[i, j] == 1:
                relative_pos = [i - com_index[0], j - com_index[1]]
                # print([i, j])
                p[:, count * 4 + 0] = -np.array(
                    [relative_pos[0] * 2 * l - 0.5 * l, relative_pos[1] * 2 * l + 0.5 * l, -0.05])
                p[:, count * 4 + 1] = -np.array(
                    [relative_pos[0] * 2 * l + 0.5 * l, relative_pos[1] * 2 * l + 0.5 * l, -0.05])
                p[:, count * 4 + 2] = -np.array(
                    [relative_pos[0] * 2 * l + 0.5 * l, relative_pos[1] * 2 * l - 0.5 * l, -0.05])
                p[:, count * 4 + 3] = -np.array(
                    [relative_pos[0] * 2 * l - 0.5 * l, relative_pos[1] * 2 * l - 0.5 * l, -0.05])
                # print(p[:, count * 4:count * 4 + 4])
                count += 1
    if verbose:
        plt.grid()
        plt.axis('equal')
        for i in range(int(n)):
            plt.plot(p[0, 4 * i:4 * i + 4], p[1, 4 * i:4 * i + 4], 'o', markersize="5")

    return p


# Specific to polytope paper
# l is the distance between two adjacent rotors
def config_to_design_matrix(config, l=0.2):
    n = np.sum(config)  # number of modules
    p = config_to_pose(config, n, l, False)

    # next, we get the A matrix
    Af = np.zeros((3, int(4*n)))
    Atau = np.zeros((3, int(4*n)))

    c = 0.001
    R = np.zeros((3, int(3 * 4 * n)))
    e3 = np.array([0, 0, 1])
    tiltAngle = np.pi/180*np.array([[-45, 45], [-45, -45], [45, -45], [45, 45]]).T
    for i in range(int(4*n)):
        R[:, 3 * i: 3 * i+3] = RotY(tiltAngle[1, i % 4]).dot(RotX(tiltAngle[0, i % 4]))
        Af[:, i] = R[:, 3 * i: 3 * i+3].dot(e3)
        Atau[:, i] = np.cross(p[:, i], R[:, 3 * i: 3 * i+3].dot(e3)) + pow(-1, i) * c * R[:, 3 * i: 3 * i+3].dot(e3)

    A = np.vstack([Af, Atau])
    # return p
    return A, p


def divide_and_conquer_hull(A):
    if A.shape[1] > 6:
        # divide
        # print("dividing for ", A.shape[1]//2)
        # print("front half")
        hull1 = divide_and_conquer_hull(A[:, :A.shape[1]//2])
        # print("back half")
        hull2 = divide_and_conquer_hull(A[:, A.shape[1]//2:])
        # conquer
        vertices = []
        # print("combining vertices")
        for v1 in list(hull1.points[hull1.vertices]):
            for v2 in list(hull2.points[hull2.vertices]):
                vertices.append(v1+v2)

        # print("number of vertices: ", len(vertices))
        S = np.array(vertices)
        hull = obtain_hull(S, 'QJ')
        return hull
    else:
        # base case
        # print("base case for ", A)
        s = getAllCombinations(A)
        S = np.array(s)
        hull = obtain_hull(S, 'QJ')
        return hull


# get binary combinations of the bases in A-matrix
def getAllCombinations(A):
    allSet = []
    num_vectors = A.shape[1]
    dimension = A.shape[0]
    for i in range(2**num_vectors):
        allSet.append(np.zeros(dimension))
        binary = bin(i)[2:].zfill(num_vectors)
        for j in range(num_vectors):
            if binary[j] == '1':
                allSet[i] = allSet[i] + A[:, j]

    return allSet


# generate a QHull object from points
def obtain_hull(points, options=''):
    return ConvexHull(points, qhull_options=options)


# checking whether a point x is in hull
def contained_in_hull(hull, x):
    # The hull is defined as all points x for which Ax + b < 0.
    # Assuming x is shape (m, d), output is boolean shape (m,).
    # A is shape (f, d) and b is shape (f, 1).
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    # print(hull_eq)
    return np.all(np.asarray(x) @ A.T + b.T <= 0, axis=1)


# populate the A-matrix
def fill_in_random_A(dim, n, identical_forces=False, force_index=None):
    rng = np.random.default_rng(int(time.time()))
    A = (rng.random((dim, n))-0.5)*20.0
    if identical_forces:
        for i in range(1, len(force_index[0])):
            A[0:int(np.ceil(dim/2)), int(force_index[0][i])] = A[0:int(np.ceil(dim/2)), int(force_index[0][0])]
            A[0:int(np.ceil(dim / 2)), int(force_index[1][i])] = A[0:int(np.ceil(dim / 2)), int(force_index[1][0])]

    return A


def satisfy_task_requirements(n, config, T, heuristic=False, opt=False):
    A, p = config_to_design_matrix(config)
    if not opt:
        hull = divide_and_conquer_hull(A)
        if contained_in_hull(hull, T).all():
            print("found the configuration!")
            return A, config, p
        else:
            S = GrowingStructure(n, config)
            while (S.get_num_modules() < n and not heuristic) or (S.get_num_modules() < n-1 and heuristic):
                S.grow(heuristic)
                print("testing {} modules".format(int(S.get_num_modules())))
                configs = S.get_config()
                for configuration in configs:
                    print("/")
                    A, p = config_to_design_matrix(configuration)
                    # print(A.dot(np.ones(int(4*np.sum(configuration)))))
                    if np.allclose(A.dot(np.ones(int(4*np.sum(configuration))))[3:], 0):
                        # print("::")
                        hull = divide_and_conquer_hull(A)
                        if contained_in_hull(hull, T).all():
                            # p = config_to_pose(configuration, S.get_num_modules(), 0.2, False)
                            print("found the configuration!")
                            return A, configuration, p

            return None
    else:
        pass
        # A_opt = matrix(A)
        # m, n = A.size
        #
        # def F(x=None, z=None):
        #     pass
        #
        # solvers.cp(F, A=A_opt, b=b)['x']
        #
        # S = GrowingStructure(n, config)
        # while (S.get_num_modules() < n and not heuristic) or (S.get_num_modules() < n-1 and heuristic):


if __name__ == "__main__":
    # profiling = []
    # ns = []
    config_nonh = np.zeros((7, 7))
    config_nonh[3, 3] = 1
    config_h = np.zeros((7, 7))
    config_h[3, 3] = 1
    # T = np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.05], [0.0, 0.5, 5.0, 0.0, 0.3, 0.0]])
    scaling = 0.45*np.eye(6)
    scaling[2, 2] *= 25
    task_requirements = []
    res_complete = []
    res_heuristic = []
    res_complete_p = []
    res_heuristic_p = []
    for i in range(1):
        # T = scaling.dot(np.random.rand(6, 20)).T
        # T[:, 2] += 1.5
        T = np.array([[1, 1, 8, 0, 0, 0],
                      [1, -1, 8, 0, 0, 0],
                      [-1, 1, 8, 0, 0, 0],
                      [-1, -2, 8, 0, 0, 0],
                      [0.5, 0.5, 8, 0, 0, -0.5],
                      [0.5, 0.5, 8, 0, 0, 0.5],
                      [-0.5, -0.5, 8, 0, 0, 0.5],
                      [-0.5, -0.5, 8, 0, 0, -0.5],
                      [-0.5, 0.5, 8, 0, 0, -0.5],
                      [-0.5, 0.5, 8, 0, 0, 0.5],
                      [0.5, -0.5, 8, 0, 0, -0.5],
                      [0.5, -0.5, 8, 0, 0, 0.5]])
        task_requirements.append(T)
        res_Acp_complete = satisfy_task_requirements(7, config_nonh, T, False)
        res_Acp_heuristic = satisfy_task_requirements(7, config_h, T, True)

        res_complete_p.append(res_Acp_complete[2])
        res_heuristic_p.append(res_Acp_heuristic[2])
        res_complete.append(res_Acp_complete[1])
        res_heuristic.append(res_Acp_heuristic[1])
