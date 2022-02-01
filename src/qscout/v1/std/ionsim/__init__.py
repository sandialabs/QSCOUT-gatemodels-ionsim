# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from numpy import abs, diag, pi, kron
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.integrate import cumtrapz

import pygsti
import pygsti.extras.interpygate as interp

from ..jaqal_gates import U_R, U_Rz, U_MS, ALL_GATES
from jaqalpaq.emulator.pygsti import AbstractNoisyNativeEmulator

from .data import get_gate_data

__version__ = '0.1.0'

class IonSimErrorModel(AbstractNoisyNativeEmulator):
    """IonSim error model of the QSCOUT native gates.

    Please type "help(IonSimErrorModel)" for more information.
    """

    # This tells AbstractNoisyNativeEmulator what gate set we're modeling:
    jaqal_gates = ALL_GATES.copy()

    def __init__(self, *args, **kwargs):
        """Builds an IonSimErrorModel instance for particular error parameters.

        
        Parameters
        ----------

        n_qubits : int
            The number of qubits to emulate (required; passed to AbstractNoisyNativeEmulator)

        model : str 
            The name of the error model to use.

        params : list of str 
            The names of the erorr parameters to use.

        v0 : dict
            The value of each error parameter, measured by its absolute difference from the value
            for an ideal gate.

        sigmas : dict
            The value of each error parameter, measured by its Gaussian width centered about v0[name].        
        
        sample : bool
            Whether to sample from the distibution of errors or average over it. 

        estimate_average : bool
            Whether to estimate the average over the error distribution, favoring efficiency
            over accuracy. 


        Notes
        -----

        Possible error parameters are the following:

            dpower12 : power error in laser tones 1 and 2 (units of W)

            dfreq1 : frequency error in laser tone 1 (units of Hz)

            dphase1 : phase error in laser tone 1 (units of rad.)

            dtime : timing error in gate duration (units of microseconds)

        R gates are Raman gates with two laser tones: 0 and 1.

        MS gates are Raman gates with three laser tones: 0, 1, and 2, where
        tone 0 is the global beam and where tones 1 and 2 are the individual beams.

        """
        
        self.set_defaults(kwargs, model=None, params=None, v0=None, sigmas=None, sample=False, estimate_average=True)

        if self.model is None or self.params is None or self.v0 is None:
            self.model = 'standard'
            self.params = ['dOmega1', 'dtime']
            self.v0 = [0, 0]
            self.sigmas = None

        # inputs = self.datadir.split('_')
        for key in self.params:
            if key not in self.v0.keys():
                self.v0[key] = 0
            if self.sigmas is not None and key not in self.sigmas.keys():
                self.sigmas[key] = 0

        # assert that number of errors specified in v0 and sigmas equals the number of inputs in the iterpygate datafile 
        assert(len(self.params) == len(self.v0))
        if self.sigmas is not None:
            assert(len(self.params) == len(self.v0))
            for sig in self.sigmas.values():
                assert(sig >= 0)

        # initialize v0_complete with requried inputs
        self.v0_complete = {
            'time_stretch': 1}

        # use v0 to overwrite and/or add to v0_complete
        for key, value in self.v0.items():
            self.v0_complete[key] = self.v0[key]

        # import gate data files
        self.gate_data = get_gate_data(self.model, self.params)

        # Pass through the balance of the parameters to AbstractNoisyNativeEmulator
        # In particular: passes the number of qubits to emulate (in args)
        super().__init__(*args, **kwargs)

    # define some functions to handle Gaussian error models

    def gaussian(self, x, scale):
        """Evaluate Gaussian distribution at x, with a standard deviation equal to scale"""
        return 1/np.sqrt(2*np.pi*scale**2)*np.exp(-x**2/(2*scale**2))

    def sample_truncnorm(self, a, b, sigma):
        """Sample from a truncated normal (Gaussian) distribution"""
        if sigma != 0:
            return truncnorm.rvs(a/sigma, b/sigma, size=1, loc=0, scale=sigma)[0] 
        else:
            return 0.0

    def estimate_chi_averaged_over_noise(self, interp_op, v0, param_ranges, sigmas):
        """Generate a process matrix, estimating an average over Gaussian distributions of noise.

        We assume the process matrix, chi(v, v0), has the functional form
        chi(v, v0) = a + b (v-v0) + c (v-v0)**2 + d (v-v0)**3, near v = v0,
        for each parameter in v.
        """
        v0 = list(v0.values())
        if sigmas is not None:
            sigmas = list(sigmas.values()) 
            for index, sigma in enumerate(sigmas):
                # print(v[index], sigma, param_ranges[index])
                assert(v0[index] - sigma >= param_ranges[index][0])
                assert(v0[index] + sigma <= param_ranges[index][1])
            interp_op.from_vector(np.array(v0) - np.array(sigmas))
            left = np.array(interp_op.to_dense())
            interp_op.from_vector(np.array(v0) + np.array(sigmas))
            right = np.array(interp_op.to_dense())
            chi = (left + right)/2.0
        else:
            interp_op.from_vector(v0)
            chi = np.array(interp_op.to_dense())
        return chi

    def average_chi_over_noise(self, interp_op, v0, param_ranges, sigmas):
        """Generate a process matrix, averaging over a single Gaussian distribution of noise"""
        v0 = list(v0.values())
        v = [vi for vi in v0]       
        if sigmas is not None:
            sigmas = list(sigmas.values())
            for index in range(len(sigmas)):
                # print(v[index], sigmas[index], param_ranges[index])
                assert(v0[index] >= param_ranges[index][0])
                assert(v0[index] <= param_ranges[index][1])
            sigs = [s for s in sigmas if s != 0.0]
            assert(len(sigs) <= 1)
            if len(sigs) == 0:
                sigma = 0.0
            else:
                sigma = sigs[0]
                index = sigmas.index(sigma)
        else:
            sigma = 0.0            
        if sigma != 0.0:
            a = - v0[index] + param_ranges[index][0]
            b = - v0[index] + param_ranges[index][1]
            max_range = min(min(abs(a), abs(b)), 4*sigma)
            xs = np.linspace(-max_range, max_range, 21)
            chis = []
            for x in xs:
                v[index] = v0[index] + x
                interp_op.from_vector(vp)
                chis += [np.array(interp_op.to_dense())]
            norm = cumtrapz(self.gaussian(xs, sigma), xs, initial=0)[-1]
            print("norm = ", norm)
            dim = len(chis[0])
            vec = []
            for i in range(dim**2):
                y0s = np.array([chi.reshape(-1)[i] for chi in chis])
                ys = y0s*self.gaussian(xs, sigma)/norm
                from matplotlib import pyplot as plt
                plt.plot(xs, y0s)
                plt.show()
                vec += [cumtrapz(ys, xs, initial=0)[-1]]
            chi = np.array(vec).reshape(dim, dim)
        else:
            interp_op.from_vector(v0)
            chi = np.array(interp_op.to_dense())
        return chi

    def sample_chi_from_noise(self, interp_op, v0, param_ranges, sigmas):
        """Generate a process matrix, sampling from Gaussian distributions of noise"""
        v0 = list(v0.values())
        v = [vi for vi in v0]
        if sigmas is not None:
            sigmas = list(sigmas.values())
            for index, sigma in enumerate(sigmas):
                if sigma != 0:
                    assert(v0[index] >= param_ranges[index][0])
                    assert(v0[index] <= param_ranges[index][1])
                    a = - v0[index] + param_ranges[index][0]
                    b = - v0[index] + param_ranges[index][1]
                    max_range = min(abs(a), abs(b))
                    v[index] = v0[index] + self.sample_truncnorm(-max_range, max_range, sigma)
                else:
                    v[index] = v0[index]

        interp_op.from_vector(v)
        chi = np.array(interp_op.to_dense())   
        return chi

    # For every gate, we need to specify a superoperator and a duration:

    def angles_in_principle_ranges(self, phi, theta):
        phi = phi if theta >= 0 else phi + np.pi
        theta = abs(theta)

        if theta > np.pi:
            phi = phi + np.pi
            theta = 2*np.pi - theta 

        phi = phi if phi >= 0 else phi + 2*np.pi
        phi = phi % (2*np.pi)
        return phi, theta

    def gateduration_R(self, q, phi, theta):
        """Compute duration of a time stretched R gate"""
        phi, theta = self.angles_in_principle_ranges(phi, theta)
        opfactory = self.gate_data['R'].opfactory
        # power0, t0, param_ranges, arg_ranges, adjust_time = opfactory.meta_data
        time_stretch = self.v0_complete['time_stretch']
        t = self.gate_data['R'].gate_process.target_duration(theta, time_stretch)
        return t

    def gate_R(self, q, phi, theta):
        """Generate a process matrix for a time stretched gate"""
        phi, theta = self.angles_in_principle_ranges(phi, theta)
        # print("... computing R gate... phi/pi, theta/pi = ", phi/np.pi, theta/np.pi)
        opfactory = self.gate_data['R'].opfactory
        param_ranges = self.gate_data['R'].param_ranges
        interp_op = opfactory.create_object((phi, theta))
        # power0, t0, param_ranges, arg_ranges, adjust_time = opfactory.meta_data
        if self.sample:
            return self.sample_chi_from_noise(interp_op, self.v0, param_ranges, self.sigmas)
        else:
            if self.estimate_average:
                return self.estimate_chi_averaged_over_noise(interp_op, self.v0, param_ranges, self.sigmas)
            else:
                return self.average_chi_over_noise(interp_op, self.v0, param_ranges, self.sigmas)

    def idle(self, q, duration):
        return diag([1, 1, 1, 1])

    # GJMS
    def gateduration_MS(self, q0, q1, phi, theta):
        """Compute duration of a time stretched MS gate"""
        phi, theta = self.angles_in_principle_ranges(phi, theta)
        opfactory = self.gate_data['MS'].opfactory
        # power0, t0, param_ranges, arg_ranges, adjust_time = opfactory.meta_data
        time_stretch = self.v0_complete['time_stretch']
        t = self.gate_data['MS'].gate_process.target_duration(theta, time_stretch)
        return t

    def gate_MS(self, q0, q1, phi, theta):
        """Generate a process matrix for a time stretched gate"""
        phi, theta = self.angles_in_principle_ranges(phi, theta)
        # print("... computing MS gate... phi/pi, theta/pi = ", phi/np.pi, theta/np.pi)
        opfactory = self.gate_data['MS'].opfactory
        param_ranges = self.gate_data['MS'].param_ranges
        interp_op = opfactory.create_object((phi, theta))
        # power0, t0, param_ranges, arg_ranges, adjust_time = opfactory.meta_data
        if self.sample:
            return self.sample_chi_from_noise(interp_op, self.v0, param_ranges, self.sigmas)
        else:
            if self.estimate_average:
                return self.estimate_chi_averaged_over_noise(interp_op, self.v0, param_ranges, self.sigmas)
            else:
                return self.average_chi_over_noise(interp_op, self.v0, param_ranges, self.sigmas)
        return

    # Rz is performed entirely in software.
    def gateduration_Rz(self, q, angle, stretch=1):
        return 0

    def gate_Rz(self, q, angle, stretch=1):
        # print("... computing Rz gate... angle/pi = ", angle/np.pi)
        return pygsti.unitary_to_pauligate(U_Rz(angle))


    # Instead of copy-pasting the above definitions, use _curry to create new methods
    # with some arguments.  None is a special argument that means: require an argument
    # in the created function and pass it through.
    C = AbstractNoisyNativeEmulator._curry

    gateduration_Rx, gate_Rx = C((None, 0.0, None), gateduration_R, gate_R)
    gateduration_Ry, gate_Ry = C((None, pi/2, None), gateduration_R, gate_R)

    gateduration_Px, gate_Px = C((None, 0.0, pi), gateduration_R, gate_R)
    gateduration_Py, gate_Py = C((None, pi/2, pi), gateduration_R, gate_R)
    gateduration_Pz, gate_Pz = C((None, pi), gateduration_Rz, gate_Rz)

    gateduration_Sx, gate_Sx = C((None, 0.0, pi/2), gateduration_R, gate_R)
    gateduration_Sy, gate_Sy = C((None, pi/2, pi/2), gateduration_R, gate_R)
    gateduration_Sz, gate_Sz = C((None, pi/2), gateduration_Rz, gate_Rz)

    gateduration_Sxd, gate_Sxd = C((None, pi, pi/2), gateduration_R, gate_R)
    gateduration_Syd, gate_Syd = C((None, 3*pi/2, pi/2), gateduration_R, gate_R)
    gateduration_Szd, gate_Szd = C((None, -1*pi/2), gateduration_Rz, gate_Rz)

    gateduration_Sxx, gate_Sxx = C((None, None, 0.0, pi/2), gateduration_MS, gate_MS)

    del C
