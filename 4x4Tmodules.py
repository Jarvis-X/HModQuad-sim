try:
    import sim
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    import matplotlib
except:
    print('--------------------------------------------------------------')
    print('Library loading failed!')
    print('')

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('figure', figsize=(7, 10))
matplotlib.rc('legend', fontsize=14)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class Robot:
    def __init__(self, frame_name, motor_names=None, control_param=None, client_id=0):
        # If there is an existing connection
        if client_id:
            self.client_id = client_id
        else:
            self.client_id = self.open_connection()

        self.motors = self._get_handlers(motor_names)

        # Robot frame
        self.frame = self._get_handler(frame_name)
        self._cal_design_matrix()

        # PID gains
        self.PID = control_param

        # logs
        self.log_p = []
        self.log_R = []
        self.log_time = []
        self.log_u = []
        self.log_rpy = []
        self.log_th = []
        self.log_tor = []
        self.log_position = []
        self.log_angles = []

    def open_connection(self):
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

        if self.client_id != -1:
            print('Robot connected')
        else:
            print('Connection failed')
        return self.client_id

    def close_connection(self):
        self.actuate([0]*len(self.motors))
        # Before closing the connection of CoppeliaSim,
        # make sure that the last command sent out had time to arrive.
        sim.simxGetPingTime(self.client_id)
        sim.simxFinish(self.client_id)  # Now close the connection of CoppeliaSim:
        print('Connection closed')

    def is_connected(self):
        c, result = sim.simxGetPingTime(self.client_id)
        # Return true if the robot is connected
        return result > 0

    def _get_handler(self, name):
        err_code, handler = sim.simxGetObjectHandle(self.client_id, name, sim.simx_opmode_blocking)
        if err_code != 0:
            print("ERROR: CANNOT GET HANDLER FOR OBJECT '{}'".format(name))
        return handler

    def _get_handlers(self, names):
        handlers = []
        if names is not None:
            for name in names:
                handler = self._get_handler(name)
                handlers.append(handler)

        return handlers

    def send_motor_velocities(self, vels):
        for motor, vel in zip(self.motors, vels):
            err_code = sim.simxSetJointTargetVelocity(self.client_id,
                                                      motor, vel, sim.simx_opmode_oneshot)

            if err_code != 0:
                print("ERROR: CANNOT SET MOTOR {} WITH VELOCITY {}".format(motor, vel))

    def set_position(self, position, relative_object=-1):
        # By default, get the position wrt the reference frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        err_code = sim.simxSetObjectPosition(self.client_id, self.frame,
                                             relative_object, position, sim.simx_opmode_oneshot)
        if err_code != 0:
            print("ERROR: CANNOT SET POSITION W.R.T. {} TO {}".format(relative_object, position))

    def set_orientation(self, orientation, relative_object=-1):
        # By default, get the position wrt the reference frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        err_code = sim.simxSetObjectOrientation(self.client_id, self.frame,
                                                relative_object, orientation, sim.simx_opmode_oneshot)
        if err_code != 0:
            print("ERROR: CANNOT SET ORIENTATION W.R.T. {} TO {}".format(relative_object, orientation))

    def sim_time(self):
        return sim.simxGetLastCmdTime(self.client_id)

    def get_position(self, relative_object=-1):
        # Get position relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, position = sim.simxGetObjectPosition(self.client_id, self.frame, relative_object, sim.simx_opmode_oneshot)
        return np.array(position)

    def get_orientation(self, relative_object=-1):
        # Get orientation relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, euler = sim.simxGetObjectOrientation(self.client_id, self.frame, relative_object, sim.simx_opmode_oneshot)
        return np.array(euler)

    def get_quaternion(self, relative_object=-1):
        # Get orientation in quaternion relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, quat = sim.simxGetObjectQuaternion(self.client_id, self.frame, relative_object, sim.simx_opmode_oneshot)
        return np.array(quat)

    def get_velocity(self, relative_object=-1):
        # Get velocity relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, velocity, omega = sim.simxGetObjectVelocity(self.client_id, self.frame, sim.simx_opmode_oneshot)
        return np.array(velocity), np.array(omega)

    def get_object_position(self, object_name):
        # Get Object position in the world frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_oneshot)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, -1, sim.simx_opmode_oneshot)
        return np.array(position)

    def get_object_relative_position(self, object_name):
        # Get Object position in the robot frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_oneshot)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, self.frame, sim.simx_opmode_oneshot)
        return np.array(position)

    def _cal_design_matrix(self):
        if len(self.motors) == 0:
            print("No propeller. Is it a magic box?")
            return

        km_over_kf = 0.0
        design_matrix = np.zeros([6, len(self.motors)])
        null_basis = np.zeros([6, 1])

        for i in range(len(self.motors)):
            motor_id = self.motors[i]
            # print(motor_id)
            res, quat_i = sim.simxGetObjectQuaternion(self.client_id, motor_id, self.frame,
                                                    sim.simx_opmode_blocking)
            # print(res)
            R_i = quat2rot(quat_i)
            res, pos_i_iterable = sim.simxGetObjectPosition(self.client_id, motor_id, self.frame,
                                                          sim.simx_opmode_blocking)
            # print(res)
            pos_i = np.array(pos_i_iterable)
            f_i = R_i.dot(np.array([0, 0, 1]))
            tau_i = np.cross(pos_i, f_i) + (-1)**(i+1)*km_over_kf*f_i
            design_matrix[:, i] = np.concatenate([f_i, tau_i])

        self.A = design_matrix

        # SVD to find F-frame
        Af = self.A[:3, :]
        w, v = np.linalg.eigh(Af.dot(Af.T))

        # pick out non-zero eigenvalues and sort them
        eigs = []
        for i in range(w.size):
            if not np.allclose(w[i], 0):
                eigs.append([w[i], v[:, i]])
        axis = sorted(eigs, reverse=True)
        # print(self.axis)

        if len(axis) == 1:
            xc = np.array([1, 0, 0])
            zaxis = axis[0][1]
            yaxis = np.cross(zaxis, xc)
            xaxis = np.cross(yaxis, zaxis)
            # rotation from structure frame to F-frame
            self.Rsf = np.vstack([xaxis, yaxis, zaxis]).T
        elif len(axis) == 2:
            zaxis = axis[0][1]
            xaxis = axis[1][1]
            yaxis = np.cross(zaxis, xaxis)
            # rotation from structure frame to F-frame
            self.Rsf = np.vstack([xaxis, yaxis, zaxis]).T
        else:
            # zaxis = axis[0][1]
            # xaxis = axis[1][1]
            # yaxis = axis[2][1]
            # for i in range(2):
            #     if not np.allclose(0, np.array([1, 0, 0]).dot(axis[i][1])):
            #         xaxis = axis[i][1]
            #         break
            # rotation from structure frame to F-frame
            self.Rsf = np.eye(3)

        print(design_matrix)
        self.controllability = np.linalg.matrix_rank(self.A)
        if self.controllability == 6:
            self.inv_A = np.linalg.pinv(self.A)
            print("Fully actuated!")
            ns = null_space(design_matrix)
            ns_std = np.inf
            for i in range(ns.shape[1]):
                if (ns[:, i] > 0).all():
                    if ns_std == np.inf:
                        print("Omnidirectional with unidirectional motors")
                    if np.std(ns[:, i]) < ns_std:
                        # find a null basis that has the smallest std
                        null_basis = ns[:, i]
                        ns_std = np.std(ns[:, i])
            self.D = np.eye(6)

        elif self.controllability == 5:
            print("5 DoF")
            # self.D = np.concatenate([[[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            #                          np.concatenate([np.zeros([4, 2]), np.eye(4)], 1)])
            self.D = np.concatenate([[[1, 0, 0, 0, 0, 0]],
                                     np.concatenate([np.zeros([4, 2]), np.eye(4)], 1)])
            # self.inv_A = np.linalg.pinv(self.A).dot(self.D)
            self.inv_A = np.linalg.pinv(self.D.dot(np.concatenate([self.Rsf.T.dot(self.A[:3]), self.A[3:, :]])))
        else:
            print("{} DoF".format(self.controllability))
            self.D = np.concatenate([np.zeros([4, 2]), np.eye(4)], 1)
            self.inv_A = np.linalg.pinv(self.D.dot(np.concatenate([self.Rsf.T.dot(self.A[:3]), self.A[3:, :]])))
            # self.inv_A = np.linalg.pinv(self.A).dot(self.D)

        self.ns = null_basis

    def set_signal(self, signal, value):
        return sim.simxSetFloatSignal(self.client_id, signal, value, sim.simx_opmode_oneshot)

    def actuate(self, u):
        thruster_names = ['f{}'.format(i+1) for i in range(len(self.motors))]
        for fi, ui in zip(thruster_names, u):
            self.set_signal(fi, ui)

    def control(self, des_pos, des_quat, rpy_d, des_vel=None, des_acc=None):
        # Get the current and desired state
        p_d = des_pos
        q_d = des_quat
        R_d = quat2rot(q_d)
        v_d, omega_d = des_vel
        a_des, alpha_des = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        if des_acc:
            a_des, alpha_des = des_acc

        # print("des vel: {}".format(des_vel))
        p = self.get_position()
        self.log_position.append(p)
        v, omega = self.get_velocity()
        rpy = self.get_orientation()
        self.log_angles.append(rpy)
        R = quat2rot(self.get_quaternion())

        # errors in position
        ep = p_d - p
        ev = v_d - v
        self.log_p.append(ep)
        self.PID.e_p_i += ep
        for i in range(len(self.PID.e_p_i)):
            if self.PID.e_p_i[i] > self.PID.cap_p_i:
                self.PID.e_p_i[i] = self.PID.cap_p_i
            elif self.PID.e_p_i[i] < -self.PID.cap_p_i:
                self.PID.e_p_i[i] = -self.PID.cap_p_i

        # errors in rotations for logs
        erpy = rpy_d - rpy
        for i in range(len(erpy)):
            if erpy[i] <= -np.pi:
                erpy[i] += 2 * np.pi
            elif erpy[i] > np.pi:
                erpy[i] -= 2 * np.pi
        self.log_rpy.append(erpy)

        # PID control for position in {W}
        kp_z, kd_z, ki_z = self.PID.kpz, self.PID.kdz, self.PID.kiz
        kp_xy, kd_xy, ki_xy = self.PID.kpxy, self.PID.kdxy, self.PID.kixy
        ar = np.concatenate([kp_xy * ep[:2] + kd_xy * ev[:2] + ki_xy * self.PID.e_p_i[:2],
                             np.array([kp_z * ep[2] + kd_z * ev[2] + ki_z * self.PID.e_p_i[2]])]) + a_des
        ar[2] += g
        f = self.PID.mass * ar

        if self.controllability != 6:
            # special cases of under-actuation
            z_d = ar / np.linalg.norm(ar)
            print(z_d)
            if self.controllability == 4:
                x_c = np.array([np.cos(rpy_d[2]), np.sin(rpy_d[2]), 0])
            else:
                _, pitch_d, yaw_d = rpy_d
                x_c = np.array([np.cos(pitch_d)*np.cos(yaw_d), np.cos(pitch_d)*np.sin(yaw_d), np.sin(pitch_d)])
            y_d = np.cross(z_d, x_c)
            y_d = y_d / np.linalg.norm(y_d)
            x_d = np.cross(y_d, z_d)
            R_d = np.vstack([x_d, y_d, z_d]).T

        eR = 0.5 * vee_map(R_d.T.dot(R.dot(self.Rsf)) - (R.dot(self.Rsf)).T.dot(R_d))
        self.log_R.append(eR)
        self.PID.e_R_i += eR
        for i in range(len(self.PID.e_R_i)):
            if self.PID.e_R_i[i] > self.PID.cap_R_i:
                self.PID.e_R_i[i] = self.PID.cap_R_i
            elif self.PID.e_R_i[i] < -self.PID.cap_R_i:
                self.PID.e_R_i[i] = -self.PID.cap_R_i

        eomega = omega_d - omega

        f = R.dot(self.Rsf).T.dot(f)

        kp_rp, kd_rp, ki_rp = self.PID.kprp, self.PID.kdrp, self.PID.kirp
        kp_y, kd_y, ki_y = self.PID.kpy, self.PID.kdy, self.PID.kiy

        aR = np.concatenate([-kp_rp * eR[:2] + kd_rp * eomega[:2] - ki_rp * self.PID.e_R_i[:2],
                             np.array([-kp_y * eR[2] + kd_y * eomega[2] - ki_y * self.PID.e_R_i[2]])]) + alpha_des

        tau = self.PID.inertia * aR

        w = np.concatenate([f, tau])
        # print("wrench: {}".format(w))
        u_crude = self.inv_A.dot(self.D.dot(w))

        if not (u_crude >= 0).all() and (self.ns > 0).all():
            min_force_div = 0
            for i in range(len(u_crude)):
                if u_crude[i] / self.ns[i] < min_force_div:
                    min_force_div = u_crude[i] / self.ns[i]

            u = u_crude - self.ns * min_force_div
            self.actuate(u.tolist())
        else:
            u = u_crude
            for i in range(len(u_crude)):
                if u_crude[i] < 0:
                    u[i] = 0
            self.actuate(u.tolist())

        # print("u: {}".format(u))
        self.log_u.append(u)

        self.log_th.append(R.dot(w[:3]))
        self.log_tor.append(w[3:])


class PID_param:
    def __init__(self, mass, inertia, KZ, KXY, KRP, KY):
        # integral stuff
        self.cap_R_i = 5.0
        self.e_R_i = np.array([0.0, 0.0, 0.0])

        self.cap_p_i = 0.5
        self.e_p_i = np.array([0.0, 0.0, 0.0])

        self.mass = mass
        self.inertia = inertia
        self.kpz, self.kdz, self.kiz = KZ
        self.kpxy, self.kdxy, self.kixy = KXY
        self.kprp, self.kdrp, self.kirp = KRP
        self.kpy, self.kdy, self.kiy = KY

def quat2rot(quat):
    # Covert a quaternion into a full three-dimensional rotation matrix.
    # Extract the values from quat
    q0 = quat[3]
    q1 = quat[0]
    q2 = quat[1]
    q3 = quat[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


# convert euler angles (roll, pitch, yaw) into quaternion (qx, qy, qz, qw)
def euler2quat(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    return np.array([sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


def vee_map(skew_s):
    # convert a skew-symmetric matrix to the corresponding array
    return np.array([skew_s[2, 1], skew_s[0, 2], skew_s[1, 0]])


# Wow, numpy does not have null space :(
def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


# Trajectory generator
def min_snap_coeff(x1, x2, v1, v2, acc1, acc2, delta_t):
    a0 = x1
    a1 = v1
    a2 = acc1/2

    # The coefficients are generated with the Mathematica script:
    # https://drive.google.com/file/d/1AJ7zPijAW1jco-QZTF0dMUIYxVBLG8-F/view?usp=sharing
    a3 = -(3*acc1*delta_t**2 - acc2*delta_t**2 + 12*delta_t*v1 + 8*delta_t*v2 + 20*x1 - 20*x2) / (2*delta_t**3)
    a4 = -(-3*acc1*delta_t**2 + 2*acc2*delta_t**2 - 16*delta_t*v1 - 14*delta_t*v2 - 30*x1 + 30*x2) / (2*delta_t**4)
    a5 = -((acc1*delta_t**2 - acc2*delta_t**2 + 6*delta_t*v1 + 6*delta_t*v2 + 12*x1 - 12*x2)/(2*delta_t**5))

    return a0, a1, a2, a3, a4, a5


# state extractor
def state_extractor(coeff, t):
    a0, a1, a2, a3, a4, a5 = coeff
    polynomial = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    derivative = a1 + 2 * a2 * t + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
    accel = 2 * a2 + 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
    return polynomial, derivative, accel


def piecewiseCoeff3D(X, Y, Z, Vx, Vy, Vz, Accx, Accy, Accz, T, num_points):
    xcoeff, ycoeff, zcoeff = [], [], []

    for i in range(num_points - 1):
        xcoeff.append(min_snap_coeff(X[i], X[i + 1], Vx[i], Vx[i + 1], Accx[i], Accx[i + 1], T[i + 1] - T[i]))
        ycoeff.append(min_snap_coeff(Y[i], Y[i + 1], Vy[i], Vy[i + 1], Accy[i], Accy[i + 1], T[i + 1] - T[i]))
        zcoeff.append(min_snap_coeff(Z[i], Z[i + 1], Vz[i], Vz[i + 1], Accz[i], Accz[i + 1], T[i + 1] - T[i]))

    return xcoeff, ycoeff, zcoeff


# state-to-state waypoint generator
def state_to_state_traj(x1, x2, v1, v2, acc1, acc2, delta_t):
    t = np.linspace(0, delta_t, 100)
    return state_extractor(min_snap_coeff(x1, x2, v1, v2, acc1, acc2, delta_t), t)


def piecewise3D (X, Y, Z, Vx, Vy, Vz, Accx, Accy, Accz, T, num_points):
    x, y, z, dx, dy, dz, ddx, ddy, ddz = [], [], [], [], [], [], [], [], []

    for i in range(num_points-1):
        xi, dxi, ddxi = state_to_state_traj(X[i], X[i+1], Vx[i], Vx[i+1], Accx[i], Accx[i+1], T[i+1] - T[i])
        yi, dyi, ddyi = state_to_state_traj(Y[i], Y[i+1], Vy[i], Vy[i+1], Accy[i], Accy[i+1], T[i+1] - T[i])
        zi, dzi, ddzi = state_to_state_traj(Z[i], Z[i+1], Vz[i], Vz[i+1], Accz[i], Accz[i+1], T[i+1] - T[i])

        x += xi.tolist()
        y += yi.tolist()
        z += zi.tolist()

        dx += dxi.tolist()
        dy += dyi.tolist()
        dz += dzi.tolist()

        ddx += ddxi.tolist()
        ddy += ddyi.tolist()
        ddz += ddzi.tolist()

    return x, y, z, dx, dy, dz, ddx, ddy, ddz


if __name__ == "__main__":
    r1 = Robot('MultiRotor',
               ['propeller{}'.format(i+1) for i in range(64)],
               PID_param(2.1, 0.2,
                         (6.0, 3.5, 0.2),
                         (6.0, 5.0, 0.2),
                         (8.0, 5.0, 0.0),
                         (15.0, 10.0, 0.05)))
    d1 = Robot('DesiredBox')
    g = 9.81

    try:
        simulation_start = time.time()
        while time.time() - simulation_start < 1:
            r1.get_position()
            d1.get_position()
            time.sleep(0.01)

        # get trajectory
        # Waypoints
        p1 = [0.0, 0.0, 0.0]
        p2 = [2.0, 0.0, 2.0]
        p3 = [0.0, 2.0, 1.5]
        p4 = [-2.5, 0.0, 1.0]
        p5 = [0.0, -2.5, 2.8]

        # Velocities
        v1 = [0.0, 0.0, 0]
        v2 = [0.0, 0.8, 0]
        v3 = [-1.0, 0.0, 0.0]
        v4 = [0.0, -1.0, 0.0]
        v5 = [0.0, 0.0, 0.0]

        # Accelerations
        acc1 = [0.0, 0.0, 0]
        acc2 = [0.0, 0.0, 0]
        acc3 = [0.0, 0.0, 0]
        acc4 = [0.0, 0.0, 0]
        acc5 = [0.0, 0.0, 0.0]

        # Waypoints of angles
        th1 = [0.0, 0.0, 0.0]
        th2 = [-0.1, 0.0, 0.5]
        th3 = [0.0, -0.2, 0.2]
        th4 = [0.1, 0.2, -0.5]
        th5 = [0.0, 0.1, 0.0]

        # Velocities of angles
        omega1 = [0.0, 0.0, 0.0]
        omega2 = [0.0, 0.0, 0.0]
        omega3 = [0.1, 0.0, -0.2]
        omega4 = [0.0, 0.0, 0.0]
        omega5 = [0.0, 0.0, 0.0]

        # Accelerations of angles
        alpha1 = [0.0, 0.0, 0.0]
        alpha2 = [0.0, 0.0, 0.0]
        alpha3 = [0.0, 0.0, 0.0]
        alpha4 = [0.0, 0.0, 0.0]
        alpha5 = [0.0, 0.0, 0.0]

        P = np.vstack((p1, p2, p3, p4, p5))
        V = np.vstack((v1, v2, v3, v4, v5))
        Acc = np.vstack((acc1, acc2, acc3, acc4, acc5))

        TH = np.vstack((th1, th2, th3, th4, th5))
        OMEGA = np.vstack((omega1, omega2, omega3, omega4, omega5))
        ALPHA = np.vstack((alpha1, alpha2, alpha3, alpha4, alpha5))

        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        Vx, Vy, Vz = V[:, 0], V[:, 1], V[:, 2]
        Accx, Accy, Accz = Acc[:, 0], Acc[:, 1], Acc[:, 2]

        rolls, pitchs, yaws = TH[:, 0], TH[:, 1], TH[:, 2]
        omegar, omegap, omegay = OMEGA[:, 0], OMEGA[:, 1], OMEGA[:, 2]
        alphar, alphap, alphay = ALPHA[:, 0], ALPHA[:, 1], ALPHA[:, 2]

        time_duration = 30.0
        T = np.array([0.0, 7.0, 14.0, 21.0, 30.0])

        x, y, z, dx, dy, dz, ddx, ddy, ddz = piecewise3D(X, Y, Z, Vx, Vy, Vz, Accx, Accy, Accz, T, 5)
        roll, pitch, yaw, droll, dpitch, dyaw, ddroll, ddpitch, ddyaw = \
            piecewise3D(rolls, pitchs, yaws, omegar, omegap, omegay, alphar, alphap, alphay, T, 5)

        for i in range(len(x)):
            time_start = time.time()
            d1.set_position(np.array([x[i], y[i], z[i]]))
            d1.log_position.append(np.array([x[i], y[i], z[i]]))
            d1.set_orientation(np.array([roll[i], pitch[i], yaw[i]]))
            d1.log_angles.append(np.array([roll[i], pitch[i], yaw[i]]))
            d1.log_time.append(time.time() - simulation_start)

            r1.control(np.array([x[i], y[i], z[i]]),
                       euler2quat(roll[i], pitch[i], yaw[i]),
                       np.array([roll[i], pitch[i], yaw[i]]),
                       (np.array([dx[i], dy[i], dz[i]]), np.array([droll[i], dpitch[i], dyaw[i]])),
                       (np.array([ddx[i], ddy[i], ddz[i]]), np.array([ddroll[i], ddpitch[i], ddyaw[i]])))
            r1.log_time.append(time.time() - simulation_start)
            # print(time.time() - time_start)
            while time.time() - time_start < time_duration/len(x):
                # d1.log_position.append(np.array([x[i], y[i], z[i]]))
                # d1.log_angles.append(np.array([roll[i], pitch[i], yaw[i]]))
                # d1.log_time.append(time.time() - simulation_start)

                r1.control(np.array([x[i], y[i], z[i]]),
                           euler2quat(roll[i], pitch[i], yaw[i]),
                           np.array([roll[i], pitch[i], yaw[i]]),
                           (np.array([dx[i], dy[i], dz[i]]), np.array([droll[i], dpitch[i], dyaw[i]])),
                           (np.array([ddx[i], ddy[i], ddz[i]]), np.array([ddroll[i], ddpitch[i], ddyaw[i]])))
                r1.log_time.append(time.time() - simulation_start)
                time.sleep(0.01)

        pos = r1.get_position()

        d1.get_position()
        d1.get_quaternion()
        d1.get_orientation()
        d1.get_velocity()
        time.sleep(0.1)

        while True:
            time_start = time.time()
            # d1.set_position(np.array([np.cos(time_start/10), np.sin(time_start/12), 1.8+0.2*np.cos(time_start/9.0)]))
            # d1.set_position(np.array([2*np.cos(time_start/2), 1.5*np.sin(time_start/2.5), 5+0.2*np.cos(time_start/3)]))
            # d1.set_orientation(np.array([0.2*np.sin(time_start/11.0), 0.3*np.cos(time_start/15.0), np.pi*0.3*np.cos(time_start/12.5)]))
            # d1.set_orientation(np.array([time_start/10.0, 0, 0]))
            # r1.control(pos, d1.get_quaternion(), d1.get_orientation(), d1.get_velocity())
            d1.log_position.append(d1.get_position())
            d1.log_angles.append(d1.get_orientation())
            d1.log_time.append(time.time() - simulation_start)

            r1.control(d1.get_position(), d1.get_quaternion(), d1.get_orientation(), d1.get_velocity())
            r1.log_time.append(time.time() - simulation_start)
            # print(time.time() - time_start)
            while time.time() - time_start < 0.05:
                time.sleep(0.001)

    except KeyboardInterrupt:
        r1.close_connection()

    log_p1 = np.array(r1.log_p)
    log_R1 = np.array(r1.log_R)
    log_rpy1 = np.array(r1.log_rpy)
    log_u1 = np.array(r1.log_u)
    log_th1 = np.array(r1.log_th)
    log_tor1 = np.array(r1.log_tor)
    log_time1 = np.array(r1.log_time)

    log_position1 = np.array(r1.log_position)
    log_angles1 = np.array(r1.log_angles)

    dlog_position1 = np.array(d1.log_position)
    dlog_angles1 = np.array(d1.log_angles)
    dlog_time1 = np.array(d1.log_time)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(231)
    # ax1.grid()
    # ax1.set_xlabel('time (s)')
    # ax1.set_ylabel('position error (m)')
    # ax1.set_title('position error v.s. time')
    # ax1.plot(log_time1, log_p1[:, 0], label='$e_x$')
    # ax1.plot(log_time1, log_p1[:, 1], label='$e_y$')
    # ax1.plot(log_time1, log_p1[:, 2], label='$e_z$')
    # ax1.legend()
    #
    # ax2 = fig.add_subplot(232)
    # ax2.grid()
    # ax2.set_xlabel('time (s)')
    # ax2.set_title('orientation error v.s. time')
    # ax2.plot(log_time1, log_R1[:, 0], label='$e_{roll}$')
    # ax2.plot(log_time1, log_R1[:, 1], label='$e_{pitch}$')
    # ax2.plot(log_time1, log_R1[:, 2], label='$e_{yaw}$')
    # ax2.legend()
    #
    # ax3 = fig.add_subplot(234)
    # ax3.grid()
    # ax3.set_xlabel('time (s)')
    # ax3.set_title('input forces v.s. time')
    # for i in range(len(r1.motors)):
    #     ax3.plot(log_time1, log_u1[:, i], label='$u_{}$'.format(i+1))
    # ax3.legend()
    #
    # ax4 = fig.add_subplot(235)
    # ax4.grid()
    # ax4.set_xlabel('time (s)')
    # ax4.set_title('rpy error v.s. time')
    # ax4.plot(log_time1, log_rpy1[:, 0], label='$e_{roll}$')
    # ax4.plot(log_time1, log_rpy1[:, 1], label='$e_{pitch}$')
    # ax4.plot(log_time1, log_rpy1[:, 2], label='$e_{yaw}$')
    # ax4.legend()
    #
    # ax5 = fig.add_subplot(233)
    # ax5.grid()
    # ax5.set_xlabel('time (s)')
    # ax5.set_title('des thrust in F v.s. time')
    # ax5.plot(log_time1, log_th1[:, 0], label='$f_x$')
    # ax5.plot(log_time1, log_th1[:, 1], label='$f_y$')
    # ax5.plot(log_time1, log_th1[:, 2], label='$f_z$')
    # ax5.legend()
    #
    # ax4 = fig.add_subplot(236)
    # ax4.grid()
    # ax4.set_xlabel('time (s)')
    # ax4.set_title('des torque in B v.s. time')
    # ax4.plot(log_time1, log_tor1[:, 0], label='$tau_{roll}$')
    # ax4.plot(log_time1, log_tor1[:, 1], label='$tau_{pitch}$')
    # ax4.plot(log_time1, log_tor1[:, 2], label='$tau_{yaw}$')
    # ax4.legend()

    # academic-style plots
    fonts = 18
    f = plt.figure()

    tstart = 0
    tmove = 0.0

    tend = 35 - tstart

    # x
    plt.subplot(6, 1, 1)
    t1 = dlog_time1
    t2 = log_time1
    plt.plot(t1 - tstart, dlog_position1[:, 0], '--r', label=r'$r_d$')

    plt.plot(t2 - tmove - tstart, log_position1[:, 0], 'b', label=r'$r$')
    plt.axis([0, tend, -3.2, 3.2])
    plt.legend(loc='upper left')
    plt.ylabel(r'$x(m)$', fontsize=fonts)

    # y
    plt.subplot(6, 1, 2)
    plt.plot(t1 - tstart, dlog_position1[:, 1], '--r')
    plt.plot(t2 - tmove - tstart, log_position1[:, 1], 'b')

    plt.axis([0, tend, -3.2, 3.2])
    # plt.legend(loc='upper left')
    plt.ylabel(r'$y(m)$', fontsize=fonts)

    # z
    plt.subplot(6, 1, 3)
    plt.plot(t1 - tstart, dlog_position1[:, 2], '--r')

    plt.plot(t2 - tmove - tstart, log_position1[:, 2], 'b', label=r'$r_d$')
    plt.axis([0, tend, -0.5, 3.5])
    # plt.legend(loc='upper left')
    plt.ylabel(r'$z(m)$', fontsize=fonts)

    # pitch
    plt.subplot(6, 1, 4)
    plt.plot(t1 - tstart, dlog_angles1[:, 1]*180/np.pi, '--r', label=r'$r$')

    plt.plot(t2 - tmove - tstart, log_angles1[:, 1]*180/np.pi, 'b', label=r'$r_d$')
    plt.axis([0, tend, -20, 20])
    # plt.legend(loc='upper left')
    plt.ylabel(r'$\phi(deg)$', fontsize=fonts)

    #
    plt.subplot(6, 1, 5)
    plt.plot(t1 - tstart, dlog_angles1[:, 0]*180/np.pi, '--r', label=r'$r_d$')

    plt.plot(t2 - tmove - tstart, log_angles1[:, 0]*180/np.pi, 'b', label=r'$r$')
    plt.axis([0, tend, -20, 20])
    # plt.legend(loc='upper left')
    plt.ylabel(r'$\theta(deg)$', fontsize=fonts)

    plt.subplot(6, 1, 6)
    plt.plot(t1 - tstart, dlog_angles1[:, 2]*180/np.pi, '--r', label=r'$r$')

    plt.plot(t2 - tmove - tstart, log_angles1[:, 2]*180/np.pi, 'b', label=r'$r_d$')
    plt.axis([0, tend, -40, 40])
    plt.ylabel(r'$\psi(deg)$', fontsize=fonts)
    plt.xlabel(r'$t(s)$', fontsize=fonts)

    f.savefig('exp_e6.pdf', bbox_inches="tight")

    f.align_ylabels()
    plt.show()

