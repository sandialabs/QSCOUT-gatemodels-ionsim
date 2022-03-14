import pygsti
import pygsti.extras.interpygate as interp

class fixed_up_interpolator():
    def __init__(self, interpolator):
        self.interpolator = interpolator
    def __call__(self, *v):
        output = self.interpolator(v)
        return output[0]

class InterpolatorData(object):
    def __init__(self, opfactory=None, target_op=None, param_ranges=None, arg_ranges=None, gate_process=None):
        self.opfactory = opfactory
        self.target_op = target_op
        self.param_ranges = param_ranges
        self.arg_ranges = arg_ranges
        self.gate_process = gate_process

class TargetOp(pygsti.modelmembers.operations.OpFactory):

    def __init__(self, dim=1, theta=None, phi=None): # BR: change dim = 4 to dim = 1 for modern pygsti
        self.process = self.create_target_gate
        pygsti.modelmembers.operations.OpFactory.__init__(self, dim, evotype="densitymx")
        self.dim = dim # should be in pygsti
        self._phi = phi
        self._theta = theta

    def create_target_unitary(self, theta, phi, sigI, sigX, sigY, sigZ):
        return sigI
    
    def create_target_gate(self, v):
        # from process_tomography import change_basis
        from pygsti.tools.basistools import change_basis
        import numpy as _np
        sigI = _np.array([[1., 0], [0, 1]], dtype='complex')
        sigX = _np.array([[0, 1], [1, 0]], dtype='complex')
        sigY = _np.array([[0, -1], [1, 0]], dtype='complex') * 1.j
        sigZ = _np.array([[1, 0], [0, -1]], dtype='complex')

        if self._phi is None:
            if self._theta is None:
                phi, theta = v
            else:
                assert(len(v) == 1)
                phi, theta = v[0], self._theta
        elif self._theta is None:
            assert(len(v) == 1)
            phi, theta = self._phi, v[0]
        else:
            phi, theta = self._phi, self._theta
        
        # print('target: phi/pi, theta/pi =', phi/_np.pi, theta/_np.pi)
        target_unitary = self.create_target_unitary(theta, phi, sigI, sigX, sigY, sigZ)
        return change_basis(_np.kron(target_unitary.conj(), target_unitary), 'col', 'pp')
    
    def create_object(self, args=None, sslbls=None):
        from pygsti.modelmembers.operations import StaticArbitraryOp # Dill tricks.
        assert(sslbls is None)
        mx = self.process([*args])
        return StaticArbitraryOp(mx)

class Gate(object):
    def __init__(self, fock_state=0, n_modes=1, n_dimensions=None,
                 b_field=7.2,
                 nominal_t0=None,
                 nominal_relative_phase=0.,
                 nominal_power=None,
                 adjust_time=None,
                 verbose=False,
                 power_scale=1.,
                 time_scale=1.,
                 cont_param_gate = False,
                 num_params = None,
                 item_shape = None,
                 aux_shape = (),
                 num_params_evaluated_as_group = 1,
                 n_qubits = None,
                 parameter_schema = None,
                 ):

        # Set the environmental parameters
        self.fock_state = fock_state
        self.n_modes = n_modes
        self.n_dimensions = n_dimensions
        self.b_field = b_field
        self.n_qubits = n_qubits

        self.nominal_t0 = nominal_t0
        self.nominal_relative_phase = nominal_relative_phase
        self.nominal_power = nominal_power
        self.adjust_time = adjust_time

        self.verbose = verbose
        self.time_scale = time_scale

        self.cont_param_gate = cont_param_gate

        self.num_params = num_params
        self.item_shape = item_shape
        self.aux_shape = aux_shape
        self.num_params_evaluated_as_group = num_params_evaluated_as_group
        self.parameter_schema = parameter_schema

    def propagate(self, initial_state=None, v=None, times=None):
        """
        :param initial_state: vector of two numbers
        :param v: relative_phase, dOmega1, dOmega2, dFreq1, dFreq2
        :param times: array of times at which to compute final state

        :return reduced_density_matrices:
        """

        import numpy as _np
        
        # Merge the fixed parameters with the input vector
        parameters = self.parameter_schema.copy()
        for param_name, param_value in zip(self.parameter_schema["vector"], v):
            parameters[param_name] = param_value

        # Special-case t/times because we sometimes need to convert it to a single-element array.
        if times is None:
            assert len(self.parameter_schema["vector"]) + 1 == len(v)
            times = [v[-1]]
        else:
            assert len(self.parameter_schema["vector"]) == len(v)

        # convert from interpygate time scale to ionsim time scale
        # print('time scale =', self.time_scale)
        times = list(_np.array(times)*self.time_scale)

        # The helper will get the parameters as local variables
        # print('parameters =', [p for p in parameters])
        # print('parameters =', [p for p in parameters.values()])
        return self.propagate_helper(initial_state, times, **parameters)

    def __call__(self, v, times=None, comm=None, basis='pp', verbose=None):
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(f'Calling process tomography as {comm.Get_rank()+1} of {comm.Get_size()} on {comm.Get_name()}.')
        processes = do_process_tomography(self.propagate, opt_args={'v':v, 'times':times},
                                          n_qubits=self.n_qubits, time_dependent=True, comm=comm, 
                                          verbose=verbose, basis=basis)
        return processes
    
    def create_process_matrix(self, v, comm=None):
        import numpy as np                                                                                                                                                                                        
        def state_to_process_mxs(state):
            return self.propagate(state, v)
        process = interp.run_process_tomography(state_to_process_mxs, n_qubits=self.n_qubits,
                                                  basis='pp', time_dependent=False, comm=comm)  # returns None on all but root processor
        return np.array(process) if (process is not None) else None
    
    def create_aux_info(self, v, comm=None):
        # relative_phase, dOmega1, dFreq1, power_noise_amp, t = v
        return [self.nominal_power, self.nominal_t0]  # matches aux_shape=() above
    
    def create_process_matrices(self, v, grouped_v, comm=None):
        import numpy as np
        assert(len(grouped_v) == 1)  # we expect a single "grouped" parameter
        times = grouped_v[0]
        def state_to_process_mxs(state):
            return self.propagate(state, v, times=times)
        processes = interp.run_process_tomography(state_to_process_mxs, n_qubits=self.n_qubits,
                                                  basis='pp', time_dependent=True, comm=comm)  # returns None on all but root processor
        return np.array(processes) if (processes is not None) else None
    
    def create_aux_infos(self, v, grouped_v, comm=None):
        times = grouped_v[0]
        return [ [self.nominal_power, self.nominal_t0] for t in times] # list elements must match aux_shape=() above
                                


class SingleQubitTargetOp(TargetOp):

    def __init__(self, theta=None, phi=None):
        TargetOp.__init__(self, 1, theta, phi) # BR: change dim = 4 to dim = 1 for modern pygsti

    def create_target_unitary(self, theta, phi, sigI, sigX, sigY, sigZ):
        import numpy as _np
        return (_np.cos(theta/2) * sigI
                - 1.j * _np.sin(theta/2) * (_np.cos(phi) * sigX + _np.sin(phi) * sigY))

class SingleQubitGate(Gate):
    def __init__(self, fock_state=0, n_modes=1, n_dimensions=3,
                 b_field=7.2,
                 nominal_t0=1.2206020602060205e-06,
                 nominal_relative_phase=0.,
                 nominal_power=25e-3,
                 verbose=False,
                 power_scale = 1.,
                 time_scale = 1.,
                 # adjust_time = None,
                 cont_param_gate = False,
                 num_params = None,
                 # process_shape = (4, 4),
                 item_shape = (4, 4),
                 aux_shape = (),
                 num_params_evaluated_as_group = 1,
                 parameter_schema = None,
                 ):

        # # Set the environmental parameters
        # if adjust_time is None:
        #     self.adjust_time = lambda theta: 1.0
        # else:
        #     self.adjust_time = adjust_time
        
        if parameter_schema is None:
            if cont_param_gate:
                parameter_schema = {
                    "power_noise_amp": 0,
                    "time_stretch": 1,
                    "dOmega2": 0,
                    "dFreq2": 0,
                    "vector": ("phi", "theta", "dOmega1", "dFreq1", "relative_phase"),
                }
            else:
                parameter_schema = {
                    "phi": 0,
                    "theta": _np.pi,
                    "relative_phase": 0,
                    "dFreq1": 0,
                    "power_noise_amp": 0,
                    "time_stretch": 1,
                    "dOmega2": 0,
                    "dFreq2": 0,
                    "vector": ("dOmega1", ),
                }
        
        # super().__init__(
        Gate.__init__(self, 
            fock_state=fock_state, n_modes=n_modes, n_dimensions=n_dimensions,
            b_field=b_field,
            nominal_t0=nominal_t0,
            nominal_relative_phase=nominal_relative_phase,
            nominal_power=nominal_power,
            # adjust_time = adjust_time,
            verbose=verbose,
            power_scale = power_scale,
            time_scale = time_scale,
            cont_param_gate = cont_param_gate,
            num_params = num_params,
            item_shape = item_shape,
            aux_shape = aux_shape,
            num_params_evaluated_as_group = num_params_evaluated_as_group,
            n_qubits = 1,
            parameter_schema = parameter_schema,
        )

    def target_duration(self, theta, time_stretch):
        """determine duration for ideal gate"""
        import numpy as _np
        adjust_time = theta/_np.pi
        t = self.nominal_t0*adjust_time*time_stretch
        return t

    def target_power(self, time_stretch):
        """determine power for ideal gate"""
        power = self.nominal_power/time_stretch
        return power

    def propagate_helper(self, initial_state, times, phi, theta, relative_phase, dFreq1, dFreq2, dOmega1, dOmega2, power_noise_amp, time_stretch, **kwargs):

        import numpy as _np

        from ionsim.physics import atom as _atom
        from ionsim.physics import laser as _laser
        from ionsim.tools import constants as _const
        from ionsim.physics import lab as _lab
        from ionsim.physics import trap as _trap
        from ionsim.physics import operator as _op

        # def adjust_power(theta):
        #     import numpy as _np
        #     return _np.sqrt(theta/(_np.pi/2))

        # define function to determine power for ideal gate
        # self.target_power = lambda time_stretch: self.nominal_power/time_stretch

        print('gate: phi/pi, theta/pi =', phi/_np.pi, theta/_np.pi)
        print("dOmega1 (W) =", dOmega1)
        print("dFreq1 (Hz) =", dFreq1)
        print("relative_phase (rad.) =", relative_phase)

        def power(t):
            import numpy as _np
            amp = power_noise_amp
            nu = 10e6
            power_0 = self.target_power(time_stretch)
            return power_0*(1 + amp*_np.sin(2*_np.pi*nu*t))
        
        relative_phase += self.nominal_relative_phase + phi
        
        # define function to determine duration of ideal gate
        # self.target_duration = lambda theta, time_stretch: self.nominal_t0*adjust_time(theta)*time_stretch

        # Define the times at which to evaluate the process
        t_stop = times[-1] + self.target_duration(theta, time_stretch)
        t_start = times[0] + self.target_duration(theta, time_stretch)
        t_steps = len(times)

        # reject negative times, set all times to zero
        if t_start < 0 or t_stop < 0:
            t_start = 0
            t_stop = 0

        # define atoms
        atom = _atom.Atom(
            'Yb171_shifted',
            b_field=self.b_field,
            keep_hyperfine=['6S1/2,0,0', '6S1/2,1,0', '6P1/2,1,1', '6P3/2,1,1'],
            exclude=['4F7/2', '5D3/2', '5D5/2', '4[3/2]1/2'],
            verbose=False,
            eliminate_states=['6P1/2', '6P3/2'],
            hyperfine_names={'zero': '6S1/2,0,0', 'one': '6S1/2,1,0',
                             'raman0':'6P1/2,1,1', 'raman1': '6P3/2,1,1'}
            )

        # define trap
        atoms = [atom]
        trap = _trap.Trap(
            atoms,
            frequencies = {'axial': 0.45e6, 'radial-a': 4.5e6, 'radial-b': 4.5e6},
            unit_vectors = [[1/_np.sqrt(2.), 1/_np.sqrt(2.), 0.0],
                            [0., 0., 1.],
                            [1/_np.sqrt(2.), -1/_np.sqrt(2.), 0.0]],
            keep_directions = ['radial-a'],
            keep_modes = [0],
            num_levels = self.n_dimensions)

        # define lasers
        shift = 0.0 
        detuning1 = 33e12 # Hz
        detuning2 = 33e12 + shift 



        laser1 = _laser.Laser().define_physical(
            power = lambda t: power(t) + dOmega1,
            power_0 = power(0),
            waist = 10e-6,
            frequency = atom.energy_difference('zero','raman0') + detuning1 + dFreq1,
            polarization = [-1/_np.sqrt(2.), 1j/_np.sqrt(2), 0.],
            nhat = [0., 0., 1.],
            phase = 0,
            states = [[atom['zero'].fine_label, atom['raman0'].fine_label],
                      [atom['zero'].fine_label, atom['raman1'].fine_label]],
            raman_states = ['zero', 'raman0'],
            )

        laser2 = _laser.Laser().define_physical(
            power = lambda t: power(t) + dOmega1,
            power_0 = power(0),
            waist = 10e-6,
            frequency = atom.energy_difference('one','raman0') + detuning2 + dFreq2,
            polarization = [-1/_np.sqrt(2.), 1j/_np.sqrt(2), 0.],
            nhat = [0., 0., -1.],
            phase = relative_phase,
            states = [[atom['one'].fine_label, atom['raman0'].fine_label],
                      [atom['one'].fine_label, atom['raman1'].fine_label]],
            raman_states = ['one', 'raman0'],
            # choose_sidebands = [0] # carrier
            )

        # define lab
        lasers = [laser1, laser2]
        lab = _lab.Lab(trap, lasers, verbose=False)

        # define initial state
        states = [[['zero'], [self.fock_state]], [['one'], [self.fock_state]]]
        lab.initialize_density_matrix(initial_state, states)

        # define operators
        gate = _op.Operator(lab, atoms, lasers).define_physical_gate(
            time_dependent_interaction=True,
            verbose=False
            )

        # use operators
        gate.evolve_density_matrix(t_start)
        gate.evolve_density_matrix(t_stop-t_start, time_steps=t_steps)

        if self.verbose:
            print('computed times:', gate.experiment_data[0])
        rhos = gate.experiment_data[1]

        # transform density matrices to the qubit's rotating frame
        num_trap_levels = self.n_dimensions
        w = (lab.hamiltonian_function(0)[num_trap_levels, num_trap_levels] - lab.hamiltonian_function(0)[0, 0]).real
        vecs = [_np.array([1, _np.exp(1j*w*t)]) for t in gate.times]
        rmats = [_np.diag(vec) for vec in vecs]
        reduced_rhos = [_np.einsum('jiki->jk', rho.reshape([2, num_trap_levels, 2, num_trap_levels])) for rho in gate.rhos]
        reduced_density_matrices = [rmat.dot(rho).dot(rmat.conj().T) for rmat, rho in zip(rmats, reduced_rhos)]
        
        # return density_matrices
        # reduced_density_matrices = [pt(dm, [2, self.n_dimensions], [1])
        #                                 for dm in density_matrices]
        
        if len(reduced_density_matrices) > 1:
            return reduced_density_matrices
        else:
            return reduced_density_matrices[0]

class MSTargetOp(TargetOp):

    def __init__(self, theta=None, phi=None):
        TargetOp.__init__(self, 2, theta, phi) #BR: change dim = 16 to dim = 2 for modern pygsti

    def create_target_unitary(self, theta, phi, sigI, sigX, sigY, sigZ):
        # from process_tomography import change_basis
        from pygsti.tools.basistools import change_basis
        import numpy as _np
        import scipy.linalg as _slin
        
        sigP = _np.cos(phi) * sigX + _np.sin(phi) * sigY
        Jphi = 0.5*(_np.kron(sigP, sigI) + _np.kron(sigI, sigP))
        return _slin.expm(-1j * theta * _np.dot(Jphi, Jphi))

class MSGate(Gate):
    def __init__(self,fock_state = 0, n_modes = 1, n_dimensions = 7, b_field = 7.2,
                    nominal_t0 = 8.888888888e-6,
                    nominal_relative_phase = 0,
                    nominal_power = 0.07569168208642624, # 0.07823222262696673 with choose_sidebands on
                    # adjust_time = None,
                    power_scale = 1.,
                    time_scale = 1.,
                    num_params_evaluated_as_group = 1,
                    cont_param_gate = False,
                    num_params = None,
                    item_shape = (16, 16),
                    aux_shape = (),
                    verbose = False,
                    parameter_schema = None):

        if parameter_schema is None:
            if cont_param_gate:
                parameter_schema = {
                    "power_noise_amp": 0,
                    "time_stretch": 1,
                    "dOmega2": 0,
                    "dFreq2": 0,
                    "dOmega3": 0,
                    "dFreq3": 0,
                    "vector": ("phi", "theta", "dOmega1", "dFreq1", "relative_phase"),
                }
            else:
                parameter_schema = {
                    "phi": 0,
                    "theta": _np.pi/2,
                    "relative_phase": 0,
                    "dFreq1": 0,
                    "power_noise_amp": 0,
                    "time_stretch": 1,
                    "dOmega2": 0,
                    "dFreq2": 0,
                    "dOmega3": 0,
                    "dFreq3": 0,
                    "vector": ("dOmega1", ),
                }
        
        #super().__init__(
        Gate.__init__(self,
            fock_state=fock_state, n_modes=n_modes, n_dimensions=n_dimensions,
            b_field=b_field,
            nominal_t0=nominal_t0,
            nominal_relative_phase=nominal_relative_phase,
            nominal_power=nominal_power,
            # adjust_time=adjust_time,
            verbose=verbose,
            power_scale = power_scale,
            time_scale = time_scale,
            cont_param_gate = cont_param_gate,
            num_params = num_params,
            item_shape = item_shape,
            aux_shape = aux_shape,
            num_params_evaluated_as_group = num_params_evaluated_as_group,
            n_qubits = 2,
            parameter_schema = parameter_schema,
        )

    def target_duration(self, theta, time_stretch):
        """function to determine duration of ideal gate"""
        import numpy as _np
        adjust_time = abs(theta)/(_np.pi/2.0)
        t = self.nominal_t0*adjust_time*time_stretch
        return t

    def target_power(self, time_stretch):
        """function to determine power for ideal gate"""
        power = self.nominal_power/time_stretch
        return power
    
    def propagate_helper(self, initial_state, times, phi, theta, relative_phase, dFreq1, dFreq2, dFreq3, dOmega1, dOmega2, dOmega3, power_noise_amp, time_stretch, **kwargs):
        
        import numpy as _np

        from ionsim.physics import atom as _atom
        from ionsim.physics import laser as _laser
        from ionsim.tools import constants as _const
        from ionsim.physics import lab as _lab
        from ionsim.physics import trap as _trap
        from ionsim.physics import operator as _op

        # def adjust_power(theta):
        #     import numpy as _np
        #     return _np.sqrt(theta/(_np.pi/2))

        # save function that determines power for ideal gate
        # self.target_power = lambda time_stretch: self.nominal_power/time_stretch

        def power(t):
            import numpy as _np
            amp = power_noise_amp
            nu = 10e6
            power_0 = self.target_power(time_stretch)
            return power_0*(1 + amp*_np.sin(2*_np.pi*nu*t))

        relative_phase += self.nominal_relative_phase + phi

        dPhase1 = relative_phase
        dPhase2 = 0
        dPhase3 = 0

        # dPhase2 += self.nominal_relative_phase
        # dPhase3 -= self.nominal_relative_phase

        # dFreq1 += self.nominal_dFreq1
        # dFreq2 += self.nominal_dFreq2
        # dFreq3 += self.nominal_dFreq3
     
        # save function that determines duration for ideal gate
        # self.target_duration = lambda theta, time_stretch: self.nominal_t0*adjust_time(theta)*time_stretch

        # sign of theta will determine handedness of trajectory in phase space
        if theta == 0:
            sign_of_theta = 1
        else:
            sign_of_theta = theta/abs(theta)

        # Define the times at which to evaluate the process
        t_stop = times[-1] + self.target_duration(abs(theta), time_stretch)
        t_start = times[0] + self.target_duration(abs(theta), time_stretch)
        t_steps = len(times)

        # reject negative times, set all times to zero
        if t_start < 0 or t_stop < 0:
            t_start = 0
            t_stop = 0

        print(f'Start time: {t_start}, Stop time: {t_stop}, Steps: {t_steps}')

        # define atoms
        atom1 = _atom.Atom(
            'Yb171_shifted',
            b_field=self.b_field,
            keep_hyperfine=['6S1/2,0,0', '6S1/2,1,0', '6P1/2,1,1', '6P3/2,1,1'],
            exclude=['4F7/2', '5D3/2', '5D5/2', '4[3/2]1/2'],
            verbose=False,
            eliminate_states=['6P1/2', '6P3/2'],
            hyperfine_names={'zero': '6S1/2,0,0', 'one': '6S1/2,1,0',
                             'raman0': '6P1/2,1,1', 'raman1': '6P3/2,1,1'}
            )
        atom2 = _atom.Atom(
            'Yb171_shifted',
            b_field=self.b_field,
            keep_hyperfine=['6S1/2,0,0', '6S1/2,1,0', '6P1/2,1,1', '6P3/2,1,1'],
            exclude=['4F7/2', '5D3/2', '5D5/2', '4[3/2]1/2'],
            verbose=False,
            eliminate_states=['6P1/2', '6P3/2'],
            hyperfine_names={'zero': '6S1/2,0,0', 'one': '6S1/2,1,0',
                             'raman0': '6P1/2,1,1', 'raman1': '6P3/2,1,1'}
            )

        # define trap
        atoms = [atom1, atom2]
        trap = _trap.Trap(
            atoms,
            frequencies = {'axial': 0.45e6, 'radial-a': 4.5e6, 'radial-b': 4.5e6},
            unit_vectors = [[1/_np.sqrt(2.),1/_np.sqrt(2.),0.0],
                            [0.,0.,1.],
                            [1/_np.sqrt(2.),-1/_np.sqrt(2.),0.0]],
            keep_directions = ['radial-a'],
            keep_modes = [1],
            num_levels = self.n_dimensions)

        # define lasers
        # a negative shift2 increases the effective laser frequencies w0 - w1 and w0 - w2
        # a positive split actually brings the two effective ms-tones closer together and further away from their respectivce target sidebands
        shift, split = 0, 0
        shift2 = shift - split/2.0
        shift3 = shift + split/2.0
        # fac = 1.0 sets the sideband detuning to zero (not including a/c stark shifts)
        dfac = sign_of_theta*0.05
        fac0 = 1 - dfac # 0.95 
        fac = 1.0 + (fac0 - 1.0)/time_stretch
        detuning1 = 33e12 + dFreq1 # Hz
        detuning2 = 33e12 + fac*trap.secular_frequencies['radial-a'][0] + shift2 + dFreq2 # Hz # fac=1 shift2=0 s resonant with red sideband qubit transition
        detuning3 = 33e12 - fac*trap.secular_frequencies['radial-a'][0] + shift3 + dFreq3 # Hz # fac=1 shift3=0 is resonant with blue sideband qubit transition

        # w1 = ER0 - ES0 + Delta
        laser1 = _laser.Laser().define_physical(
            power = lambda t: power(t) + dOmega1,
            power_0 = power(0),
            waist = 10e-6,
            frequency = atom1.energy_difference('zero', 'raman0') + detuning1,
            polarization = [-1/_np.sqrt(2.), 1j/_np.sqrt(2), 0.],
            nhat = [0., 0., 1.],
            phase = dPhase1,
            states = [[atom1['zero'].fine_label, atom1['raman0'].fine_label],
                      [atom1['zero'].fine_label, atom1['raman1'].fine_label]],
            raman_states = ['zero', 'raman0'],
            # power_modulation = power_modulation
            )

        # w2 = ER0 - ES1 + Delta + v + shift ; near red sideband
        laser2 = _laser.Laser().define_physical(
            power = lambda t: power(t) + dOmega2,
            power_0 = power(0),
            waist = 10e-6,
            frequency = atom1.energy_difference('one', 'raman0') + detuning2,
            polarization = [-1/_np.sqrt(2.), 1j/_np.sqrt(2), 0.],
            nhat = [0., 0., -1.],
            phase = dPhase2,
            states = [[atom1['one'].fine_label, atom1['raman0'].fine_label],
                      [atom1['one'].fine_label, atom1['raman1'].fine_label]],
            raman_states = ['one', 'raman0'],
            choose_sidebands = [-1], # red sideband,
            # power_modulation = power_modulation
            )

        # w3 = ER0 - ES1 + Delta - v + shift ; near blue sideband
        laser3 = _laser.Laser().define_physical(
            power = lambda t: power(t) + dOmega3,
            power_0 = power(0),
            waist = 10e-6,
            frequency = atom1.energy_difference('one', 'raman0') + detuning3,
            polarization = [-1/_np.sqrt(2.), 1j/_np.sqrt(2), 0.],
            nhat = [0., 0., -1.],
            phase = dPhase3,
            states = [[atom1['one'].fine_label, atom1['raman0'].fine_label],
                      [atom1['one'].fine_label, atom1['raman1'].fine_label]],
            raman_states = ['one', 'raman0'],
            choose_sidebands = [1], # blue sideband,
            # power_modulation = power_modulation
            )

        # define lab
        lasers = [laser1, laser2, laser3]
        lab = _lab.Lab(trap, lasers, verbose=False)

        # define initial state
        coefs = initial_state
        states = [[['zero', 'zero'], [self.fock_state]],
                  [['zero', 'one'], [self.fock_state]], 
                  [['one', 'zero'], [self.fock_state]],
                  [['one', 'one'], [self.fock_state]]]
        lab.initialize_density_matrix(coefs, states)

        # define operators
        ms_gate = _op.Operator(lab, atoms, lasers).define_physical_gate(
            time_dependent_interaction=True,
            verbose=False
            )

        # use operators
        ms_gate.evolve_density_matrix(t_start)
        ms_gate.evolve_density_matrix(t_stop-t_start, time_steps=t_steps)

        print('computed times:', ms_gate.experiment_data[0])
        rhos = ms_gate.experiment_data[1]

        # transform density matrices to the qubit's rotating frame
        num_trap_levels = self.n_dimensions
        w = -1.0*(lab.hamiltonian_function(0)[num_trap_levels, num_trap_levels] - lab.hamiltonian_function(0)[0, 0]).real
        vecs = [_np.array([1, _np.exp(1j*w*t), _np.exp(1j*w*t), _np.exp(2*1j*w*t)]) for t in ms_gate.times]
        rmats = [_np.diag(vec) for vec in vecs]
        reduced_rhos = [_np.einsum('jiki->jk', rho.reshape([4, num_trap_levels, 4, num_trap_levels])) for rho in ms_gate.rhos]
        reduced_density_matrices = [rmat.dot(rho).dot(rmat.conj().T) for rmat, rho in zip(rmats, reduced_rhos)]

        # reduced_density_matrices = [pt(dm, [2, 2, self.n_dimensions], [2]) for dm in density_matrices]

        if len(reduced_density_matrices) > 1:
            return reduced_density_matrices
        else:
            return reduced_density_matrices[0]
