from tensorforce.environments import Environment
from shenfun import *
from simulation_base.utils import get_indecies

import matplotlib.pyplot as plt
import sympy
import sys
import os

import sympy

BASE_DT = 0.05
NUM_DT_BETWEEN_ACTIONS = 16
MAX_EPISODE_TIMESTEPS = 250
X_SHAPE = 20
Y_SHAPE = 20
NUM_STATE_POINTS_X = 8
NUM_STATE_POINTS_Y = 8
NUM_ACTIONS = 10
NUM_PREV_TIMESTEPS_STATE = 4

x, y, t = sympy.symbols('x,y,t', real=True)

RB_CONFIG = {
    'N': (X_SHAPE, Y_SHAPE),
    'Ra': 500,
    "Pr": 0.71,
    'dt': BASE_DT,
    'filename': 'RB100',
    'conv': 1,
    'modplot': 100,
    'modsave': 50,
        'bcT': (sympy.sin((t+x)), 0),
    'family': 'C',
    'quad': 'GC'
}



class RayleighBenardEnvironment(Environment):
    def __init__(self, num_dt_between_actions=NUM_DT_BETWEEN_ACTIONS, max_episode_timesteps=MAX_EPISODE_TIMESTEPS,
        num_actions=NUM_ACTIONS, RB_config=RB_CONFIG):
        super().__init__()

        self.num_dt_between_actions = num_dt_between_actions
        self.max_episode_timesteps_value = max_episode_timesteps
        self.num_actions = num_actions

        self.RB_config = RB_config

        self.reset()

    def __get_state(self):
        return self.RB.get_state()

    def __get_reward(self):
        return self.RB.get_reward()

    def reset(self):
        self.time_step = 0

        self.RB = RayleighBenard(**self.RB_config)
        self.RB.initialize(rand = 0.01)
        self.RB.assemble()
        

        #move simulation forward NUM_PREV_TIMESTEPS_STATE steps in order to have complete state at initial actions
        for _ in range(NUM_PREV_TIMESTEPS_STATE):
            actions=np.ones(NUM_ACTIONS)*2
            actions = self.__expand_actions_shape(actions)

            self.RB.solve(num_timesteps = self.num_dt_between_actions,
                actions = actions)

        self.state = self.__get_state()
        return self.state

    def max_episode_timesteps(self):
        return self.max_episode_timesteps_value

    def __normalize_actions(self, actions):
        """
        Make sure that the mean of the actions are 2 in order to have a stable
        Ra
        """
        actions_mean = np.mean(actions)
        diff = 2-actions_mean
        new_actions = actions + diff

        return new_actions

    def __expand_actions_shape(self, actions):
        """
        Make sure that an action will be applied over several indecies in order
        to decrease the action space
        """
        boundary_points = self.RB.N[1]
        action_points = len(actions)
        msg = f"N[1] must be divisible by num_actions, currently: {boundary_points} and {action_points}"
        assert boundary_points % action_points == 0, msg

        repeat_factor = int(boundary_points/action_points)

        new_actions = [action for action in actions for _ in range(repeat_factor)]

        return np.array(new_actions)


    def execute(self, actions, output = False):
        self.time_step += 1
        print(self.time_step)
        actions = self.__expand_actions_shape(actions)
        actions = self.__normalize_actions(actions)
        self.RB.solve(num_timesteps = self.num_dt_between_actions,
            actions = actions)

        new_state = self.__get_state()
        reward = self.__get_reward()

        terminal = self.time_step == self.max_episode_timesteps()

        if output:
            print(f"terminal: {terminal}, reward: {reward}")

        return new_state, terminal, reward

    def actions(self):
        return dict(type = "int", shape = self.num_actions,
            num_values = 2)

    def states(self):
        return dict(type = "float", shape = (NUM_PREV_TIMESTEPS_STATE*NUM_STATE_POINTS_X*NUM_STATE_POINTS_Y*3, ))

x, y, tt = sympy.symbols('x,y,t', real=True)

class RayleighBenard(object):
    def __init__(self, N=(32, 32), L=(2, 2*np.pi), Ra=100., Pr=0.7, dt=0.1,
                 bcT=(1, 2), conv=0, modplot=100, modsave=1e8, filename='RB',
                 family='C', quad='GC', num_actions = 4, num_states = 4):


        self.nu = np.sqrt(Pr/Ra)
        self.kappa = 1./np.sqrt(Pr*Ra)
        self.dt = dt
        self.N = np.array(N)
        self.L = np.array(L)
        self.conv = conv
        self.modplot = modplot
        self.modsave = modsave
        self.bcT = bcT
        self.t = 0
        self.family = family
        self.quad = quad

        self.a = (8./15., 5./12., 3./4.)
        self.b = (0.0, -17./60., -5./12.)
        self.c = (0., 8./15., 2./3., 1)

        # Regular spaces
        self.sol = chebyshev if family == 'C' else legendre
        self.B0 = FunctionSpace(N[0], family, quad=quad, bc='Biharmonic')
        self.D0 = FunctionSpace(N[0], family, quad=quad, bc=(0, 0))
        self.C0 = FunctionSpace(N[0], family, quad=quad)
        self.T0 = FunctionSpace(N[0], family, quad=quad, bc=bcT)
        self.F1 = FunctionSpace(N[1], 'F', dtype='d')
        self.D00 = FunctionSpace(N[0], family, quad=quad, bc=(0, 0))  # Streamwise velocity, not to be in tensorproductspace

        # Regular tensor product spaces
        self.TB = TensorProductSpace(comm, (self.B0, self.F1)) # Wall-normal velocity
        self.TD = TensorProductSpace(comm, (self.D0, self.F1)) # Streamwise velocity
        self.TC = TensorProductSpace(comm, (self.C0, self.F1)) # No bc
        self.TT = TensorProductSpace(comm, (self.T0, self.F1)) # Temperature
        self.BD = VectorSpace([self.TB, self.TD])  # Velocity vector
        self.CD = VectorSpace([self.TD, self.TD])  # Convection vector

        # Padded for dealiasing
        self.TBp = self.TB.get_dealiased((1.5, 1.5))
        self.TDp = self.TD.get_dealiased((1.5, 1.5))
        self.TCp = self.TC.get_dealiased((1.5, 1.5))
        self.TTp = self.TT.get_dealiased((1.5, 1.5))
        #self.TBp = self.TB
        #self.TDp = self.TD
        #self.TCp = self.TC
        #self.TTp = self.TT
        self.BDp = VectorSpace([self.TBp, self.TDp])  # Velocity vector

        self.u_ = Function(self.BD)
        self.ub = Array(self.BD)
        self.up = Array(self.BDp)
        self.w0 = Function(self.TC).v
        self.w1 = Function(self.TC).v
        self.uT_ = Function(self.BD)
        self.T_ = Function(self.TT)
        self.T_b = Array(self.TT)
        self.T_p = Array(self.TTp)
        self.T_1 = Function(self.TT)
        self.rhs_u = Function(self.CD)   # Not important which space, just storage
        self.rhs_T = Function(self.CD)
        self.u00 = Function(self.D00)
        self.b0 = np.zeros((2,)+self.u00.shape)

        self.dudxp = Array(self.TDp)
        self.dudyp = Array(self.TBp)
        self.dvdxp = Array(self.TCp)
        self.dvdyp = Array(self.TDp)

        # self.file_u = ShenfunFile('_'.join((filename, 'U')), self.BD, backend='hdf5', mode='w', uniform=True)
        # self.file_T = ShenfunFile('_'.join((filename, 'T')), self.TT, backend='hdf5', mode='w', uniform=True)

        self.mask = self.TB.get_mask_nyquist()
        self.K = self.TB.local_wavenumbers(scaled=True)
        self.X = self.TD.local_mesh(True)

        self.H_ = Function(self.CD)  # convection
        self.H_1 = Function(self.CD)
        self.H_2 = Function(self.CD)

        self.curl = Function(self.TCp)
        self.wa = Array(self.TCp)

        self.temperature = dict()
        self.u = dict()
        self.actions_list = dict()
        self.nusselt_list = dict()

        self.states_list = list()


    def initialize(self, rand=0.01):
        X = self.TB.local_mesh(True)
        funT = 1 if self.bcT[0] == 1 else 2
        fun = {1: 1,
               2: (0.9+0.1*np.sin(2*X[1]))}[funT]
        self.T_b[:] = 0.5*(1-X[0])*fun+rand*np.random.randn(*self.T_b.shape)*(1-X[0])*(1+X[0])
        self.T_ = self.T_b.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        self.T_1[:] = self.T_

    def assemble(self):
        u = TrialFunction(self.TB)
        v = TestFunction(self.TB)
        p = TrialFunction(self.TT)
        q = TestFunction(self.TT)
        nu = self.nu
        dt = self.dt
        kappa = self.kappa

        # Note that we are here assembling implicit left hand side matrices,
        # as well as matrices that can be used to assemble the right hande side
        # much faster through matrix-vector products

        a, b = self.a, self.b
        self.solver = []
        for rk in range(3):
            mats = inner(v, div(grad(u)) - ((a[rk]+b[rk])*nu*dt/2.)*div(grad(div(grad(u)))))
            self.solver.append(self.sol.la.Biharmonic(*mats))

        self.solverT = []
        self.lhs_mat = []
        for rk in range(3):
            matsT = inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*p - div(grad(p)))
            self.lhs_mat.append(extract_bc_matrices([matsT]))
            self.solverT.append(self.sol.la.Helmholtz(*matsT))

        u0 = TrialFunction(self.D00)
        v0 = TestFunction(self.D00)
        self.solver0 = []
        for rk in range(3):
            mats0 = inner(v0, 2./(nu*(a[rk]+b[rk])*dt)*u0 - div(grad(u0)))
            self.solver0.append(self.sol.la.Helmholtz(*mats0))

        self.B_DD = inner(TestFunction(self.TD), TrialFunction(self.TD))
        self.C_DB = inner(Dx(TrialFunction(self.TB), 0, 1), TestFunction(self.TD))


        #fast
        u = TrialFunction(self.TB)
        v = TestFunction(self.TB)
        sv = TrialFunction(self.CD)
        p = TrialFunction(self.TT)
        q = TestFunction(self.TT)
        nu = self.nu
        dt = self.dt
        kappa = self.kappa
        a = self.a
        b = self.b

        # Assemble matrices that are used to compute the right hande side
        # through matrix-vector products

        self.mats_u = []
        self.mats_uT = []
        self.mats_conv = []
        self.mats_rhs_T = []
        self.rhs_mat = []
        for rk in range(3):
            self.mats_u.append(inner(v, div(grad(u)) + (nu*(a[rk]+b[rk])*dt/2.)*div(grad(div(grad(u))))))
            self.mats_rhs_T.append(inner(q, 2./(kappa*(a[rk]+b[rk])*dt)*p + div(grad(p))))
            self.rhs_mat.append(extract_bc_matrices([self.mats_rhs_T[-1]]))

        #self.mats_uT = inner(v, div(grad(p)))
        self.mats_uT = inner(v, Dx(p, 1, 2))
        self.mat_conv = inner(v, (Dx(Dx(sv[1], 0, 1), 1, 1) - Dx(sv[0], 1, 2)))

        uv = TrialFunction(self.BD)
        self.mats_div_uT = inner(q, div(uv))

        vc = TestFunction(self.TC)
        uc = TrialFunction(self.TC)
        self.A_TC = inner(vc, uc)
        self.curl_rhs = inner(vc, Dx(uv[1], 0, 1) - Dx(uv[0], 1, 1))

        vd = TestFunction(self.TD)
        ud = TrialFunction(self.TD)
        self.A_TD = inner(vd, ud)
        self.CDB = inner(vd, Dx(u, 0, 1))
        self.CTD = inner(vc, Dx(ud, 0, 1))

    def end_of_tstep(self):
        self.T_1[:] = self.T_
        self.rhs_T[:] = 0
        self.rhs_u[:] = 0

    def update_bc(self, t):
        # Update the two bases with time-dependent bcs.
        #self.T0.bc.update_bcs_time(t)
        self.TTp.bases[0].bc.update_bcs_time(t)

    def compute_curl(self, u):
        self.w1[:] = 0
        for mat in self.curl_rhs:
            self.w1 += mat.matvec(u, self.w0)
        curl = self.A_TC.solve(self.w1, self.curl)
        curl.mask_nyquist(self.mask)
        return curl.backward(self.wa)

    def convection(self, u, H):
        up = self.BDp.backward(u, self.up)
        if self.conv == 0:
            #dudxp = project(Dx(u[0], 0, 1), self.TDp).backward(self.dudxp)
            self.w0 = self.CDB.matvec(u[0], self.w0)
            self.w0 = self.A_TD.solve(self.w0)
            dudxp = self.TDp.backward(self.w0, self.dudxp)
            #dudyp = project(Dx(u[0], 1, 1), self.TBp).backward(self.dudyp)
            dudyp = self.TBp.backward(1j*self.K[1]*u[0], self.dudyp)
            #dvdxp = project(Dx(u[1], 0, 1), self.TCp).backward(self.dvdxp)
            self.w0 = self.CTD.matvec(u[1], self.w0)
            self.w0 = self.A_TC.solve(self.w0)
            dvdxp = self.TCp.backward(self.w0, self.dvdxp)
            #dvdyp = project(Dx(u[1], 1, 1), self.TDp).backward(self.dvdyp)
            dvdyp = self.TDp.backward(1j*self.K[1]*u[1], self.dvdyp)
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp, H[1])
        elif self.conv == 1:
            curl = self.compute_curl(u)
            H[0] = self.TDp.forward(-curl*up[1])
            H[1] = self.TDp.forward(curl*up[0])
        H.mask_nyquist(self.mask)
        return H

    def compute_rhs_u(self, rhs, rk):
        a = self.a[rk]
        b = self.b[rk]
        H_ = self.convection(self.u_, self.H_)
        rhs[1] = 0
        for mat in self.mats_u[rk]:
            rhs[1] += mat.matvec(self.u_[0], self.w0)
        self.w1[:] = 0
        for mat in self.mat_conv:
            self.w1 += mat.matvec(H_, self.w0)
        for mat in self.mats_uT:
            self.w1 += mat.matvec(self.T_, self.w0)
        rhs[1] += a*self.dt*self.w1
        rhs[1] += b*self.dt*rhs[0]
        rhs[0] = self.w1
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_rhs_T(self, rhs, rk):
        a = self.a[rk]
        b = self.b[rk]
        rhs[1] = 0
        for mat in self.mats_rhs_T[rk]:
            rhs[1] += mat.matvec(self.T_, self.w0) #same as rhs = inner(q, (2./kappa/dt)*Expr(T_1) + div(grad(T_1)), output_array=rhs)

        # The following two are equal as long as the bcs is constant
        # For varying bcs they need to be included
        if isinstance(self.bcT[0], sympy.Expr):
            rhs[1] -= self.lhs_mat[rk][0].matvec(self.T_1, self.w0)
            rhs[1] += self.rhs_mat[rk][0].matvec(self.T_, self.w1)


        up = self.BDp.backward(self.u_, self.up)
        T_p = self.TTp.backward(self.T_, self.T_p)
        uT_ = self.BDp.forward(up*T_p, self.uT_)

        self.w1[:] = 0
        for mat in self.mats_div_uT:
            self.w1 += mat.matvec(uT_, self.w0) # same as rhs -= (2./self.kappa)*inner(q, div(uT_))
        rhs[1] -= (2.*a/self.kappa/(a+b))*self.w1
        rhs[1] -= (2.*b/self.kappa/(a+b))*rhs[0]
        rhs[0] = self.w1
        rhs.mask_nyquist(self.mask)
        return rhs

    def compute_v(self, u, rk):
        v0 = TestFunction(self.D00)
        if comm.Get_rank() == 0:
            self.u00[:] = u[1, :, 0].real
            w00 = np.zeros_like(self.u00)
        #dudx_hat = project(Dx(u[0], 0, 1), self.TD)
        #with np.errstate(divide='ignore'):
        #    u[1] = 1j * dudx_hat / self.K[1]
        dudx_hat = self.C_DB.matvec(u[0], self.w0)
        with np.errstate(divide='ignore'):
            dudx_hat = 1j * dudx_hat / self.K[1]
        u[1] = self.B_DD.solve(dudx_hat, u=u[1])

        # Still have to compute for wavenumber = 0
        if comm.Get_rank() == 0:
            a, b = self.a[rk], self.b[rk]
            self.b0[1] = inner(v0, 2./(self.nu*(a+b)*self.dt)*Expr(self.u00)+div(grad(self.u00)))
            w00 = inner(v0, self.H_[1, :, 0], output_array=w00)
            self.b0[1] -= (2.*a/self.nu/(a+b))*w00
            self.b0[1] -= (2.*b/self.nu/(a+b))*self.b0[0]
            self.u00 = self.solver0[rk](self.u00, self.b0[1])
            u[1, :, 0] = self.u00
            self.b0[0] = w00
        return u

    def init_plots(self, plot = False):
        ub = self.u_.backward(self.ub)
        T_b = self.T_.backward(self.T_b)
        if comm.Get_rank() == 0 and plot:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.quiver(self.X[1], self.X[0], ub[1], ub[0], pivot='mid', scale=0.01)
            plt.draw()
            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1], self.X[0], T_b, 100)
            plt.draw()
            plt.pause(1e-6)

    def set_state_current_timestep(self):
        shape = self.T_b.shape
        indecies = get_indecies(shape, NUM_STATE_POINTS_X, NUM_STATE_POINTS_Y)

        tmp_T = self.T_b
        tmp_u1 = self.ub[0]
        tmp_u2 = self.ub[1]

        #state for current time step
        current_state = list()
        for arr in [tmp_T, tmp_u1, tmp_u2]:
            for ind in indecies:
                x, y = ind[0], ind[1]
                current_state.append(arr[x, y])

        current_state = np.array(current_state)
        self.states_list.append(current_state)

    def get_state(self):
        state = np.array(self.states_list[-NUM_PREV_TIMESTEPS_STATE:]).flatten()
        return state

    def get_reward(self):
        dT = project(Dx(self.T_, 0, 1), self.TC)
        conduction = inner(1, abs(dT.backward()))

        convection_values = self.convection(self.u_, self.H_).backward()
        convection = inner((1, 1), abs(convection_values))

        nusselt = np.sum(convection)/np.sum(conduction)
        self.nusselt_list[self.t] = nusselt

        return -nusselt

    def plot(self, t, tstep):
        if tstep % self.modplot == 0:
            ub = self.u_.backward(self.ub)
            e0 = dx(ub[0]*ub[0])
            e1 = dx(ub[1]*ub[1])
            T_b = self.T_.backward(self.T_b)
            e2 = inner(1, T_b*T_b)
            div_u = project(div(self.u_), self.TD).backward()
            e3 = dx(div_u*div_u)
            if comm.Get_rank() == 0:
                print("Time %2.5f Energy %2.6e %2.6e %2.6e div %2.6e" %(t, e0, e1, e2, e3))

                plt.figure(1)
                self.im1.set_UVC(ub[1], ub[0])
                self.im1.scale = np.linalg.norm(ub[1])
                plt.pause(1e-6)
                plt.figure(2)
                self.im2.ax.clear()
                self.im2.ax.contourf(self.X[1], self.X[0], T_b, 100)
                self.im2.autoscale()
                plt.pause(1e-6)

    def tofile(self, tstep):
        ub = self.u_.backward(self.ub, uniform=True)
        T_b = self.T_.backward(uniform=True)
        # self.file_u.write(tstep, {'u': [ub]}, as_scalar=True)
        # self.file_T.write(tstep, {'T': [T_b]})

    #@profile
    def solve(self, num_timesteps, actions):

        for _ in range(num_timesteps):
            # Fix the new bcs in the solutions. Don't have to fix padded T_p because it is assembled from T_1 and T_2
            self.T0.bc.update_bcs(bc = (actions, 0))

            for rk in range(3):
                self.T0.bc.set_tensor_bcs(this_base = self.T0, T = self.TT)
                # self.T0.bc.set_boundary_dofs(self.T_, True)
                self.update_bc(self.t+self.dt*self.c[rk+1]) # Update bc for next step
                # self.T0.bc.set_boundary_dofs(self.T_1, True) # T_1 holds next step bc

                rhs_u = self.compute_rhs_u(self.rhs_u, rk)
                self.u_[0] = self.solver[rk](self.u_[0], rhs_u[1])
                if comm.Get_rank() == 0:
                    self.u_[0, :, 0] = 0
                u_ = self.compute_v(self.u_, rk)
                u_.mask_nyquist(self.mask)
                rhs_T = self.compute_rhs_T(self.rhs_T, rk)
                T_ = self.solverT[rk](self.T_, rhs_T[1])
                T_.mask_nyquist(self.mask)

                self.T_1 = T_

            self.t += self.dt
            self.end_of_tstep()

        ub = self.u_.backward(self.ub, kind = "uniform")
        T_b = self.T_.backward(kind = "uniform")

        self.temperature[self.t] = T_b
        self.u[self.t] = ub
        self.actions_list[self.t] = actions
        self.set_state_current_timestep() #timestep for current timestep (complete agent state is previous N steps merged)

    def save_to_file(self, folderpath = None, output=True):

        time = np.array(list(self.temperature.keys()))
        temp = np.array(list(self.temperature.values()))
        u = np.array(list(self.u.values()))
        actions= np.array(list(self.actions_list.values()))
        nusselt = np.array(list(self.nusselt_list.values()))

        if folderpath is not None:
            np.save(file = os.path.join(folderpath, "u.npy"), arr = u)
            np.save(file = os.path.join(folderpath,"time.npy"), arr = time)
            np.save(file = os.path.join(folderpath, "temp.npy"), arr = temp)
            np.save(file = os.path.join(folderpath, "actions.npy"), arr = actions)
            np.save(file = os.path.join(folderpath, "nusselt.npy"), arr = nusselt)

            if output:
                print("saved files to folder: {}".format(folderpath))

        else:
            np.save(file = "u", arr = u)
            np.save(file = "time", arr = time)
            np.save(file = "temp", arr = temp)
