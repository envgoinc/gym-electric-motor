from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import RCVoltageSupply
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gym_electric_motor.reference_generators import ConstReferenceGenerator
from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper
from gym_electric_motor.physical_systems.converters import ContB6BridgeConverter
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym

class CurrentVectorProcessor(PhysicalSystemWrapper):
    """Adds an ``i_abs`` state to the systems state vector that is the root of the squared sum of the currents ``i_sd`` and `` i_sq``."""

    def __init__(self, physical_system=None):
        self._i_sd_idx = None
        self._i_sq_idx = None
        self._i_a_idx = None
        self._i_b_idx = None
        self._i_c_idx = None
        self._u_sd_idx = None
        self._u_sq_idx = None
        self._u_a_idx = None
        self._u_b_idx = None
        self._u_c_idx = None
        self._omega_idx = None
        self._torque_idx = None
        super().__init__(physical_system=physical_system)

    def set_physical_system(self, physical_system):
        """Writes basic data like state indices to locals and finally initializes the instance.

        Args:
            physical_system(PhysicalSystem): The inner physical system of the processor.

        Returns:
            this: This instance.
        """
        super().set_physical_system(physical_system)
        # Define the new state space as concatenation of the old state space and [-1,1] for i_abs
        low = np.concatenate((physical_system.state_space.low, [-1.]))
        high = np.concatenate((physical_system.state_space.high, [1.]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)

        # Set the new limits /  nominal values of the state vector
        self._i_sq_idx = self._physical_system.state_names.index('i_sq')
        self._i_sd_idx = self._physical_system.state_names.index('i_sd')
        self._i_a_idx = self._physical_system.state_names.index('i_a')
        self._i_b_idx = self._physical_system.state_names.index('i_b')
        self._i_c_idx = self._physical_system.state_names.index('i_c')
        self._u_sd_idx = self._physical_system.state_names.index('u_sd')
        self._u_sq_idx = self._physical_system.state_names.index('u_sq')
        self._u_a_idx = self._physical_system.state_names.index('u_a')
        self._u_b_idx = self._physical_system.state_names.index('u_b')
        self._u_c_idx = self._physical_system.state_names.index('u_c')
        self._omega_idx = self._physical_system.state_names.index('omega')
        self._torque_idx = self._physical_system.state_names.index('torque')
        self.i_sd_limit = physical_system.limits[self._i_sd_idx]
        self.i_sq_limit = physical_system.limits[self._i_sq_idx]
        self.i_a_limit = physical_system.limits[self._i_a_idx]
        self.i_b_limit = physical_system.limits[self._i_b_idx]
        self.i_c_limit = physical_system.limits[self._i_c_idx]
        self.u_sd_limit = physical_system.limits[self._u_sd_idx]
        self.u_sq_limit = physical_system.limits[self._u_sq_idx]
        self.u_a_limit = physical_system.limits[self._u_a_idx]
        self.u_b_limit = physical_system.limits[self._u_b_idx]
        self.u_c_limit = physical_system.limits[self._u_c_idx]
        self.omega_limit = physical_system.limits[self._omega_idx]
        self.torque_limit = physical_system.limits[self._torque_idx]
        current_limit = np.sqrt((physical_system.limits[self._i_sd_idx]**2 + physical_system.limits[self._i_sq_idx]**2) / 2)
        current_nominal_value = np.sqrt((physical_system.nominal_state[self._i_sd_idx]**2 + physical_system.nominal_state[self._i_sq_idx]**2)/2)
        self._limits = np.concatenate((physical_system.limits, [current_limit]))
        self._nominal_state = np.concatenate((physical_system.nominal_state, [current_nominal_value]))

        # Append the new state to the state name vector and the state positions dictionary
        self._state_names = physical_system.state_names + ['i_abs']
        self._state_positions = physical_system.state_positions.copy()
        self._state_positions['i_abs'] = self._state_names.index('i_abs')
        return self

    def reset(self):
        """Resets this instance and the inner system

        Returns:
            np.ndarray: The initial state after the reset."""
        state = self._physical_system.reset()
        return np.concatenate((state, [self._get_current_abs(state)]))

    def simulate(self, action):
        """Simulates one step of the system."""
        state = self._physical_system.simulate(action)
        return np.concatenate((state, [self._get_current_abs(state)]))

    def _get_current_abs(self, state):
        """Calculates the root sum of the squared currents from the state

        Args:
            state(numpy.ndarray[float]): The state of the inner system.

        Returns:
            float: The rms of the currents of the state.
        """
        self.i_sd=state[self._i_sd_idx] * self.i_sd_limit
        self.i_sq=state[self._i_sq_idx] * self.i_sq_limit
        self.i_a=state[self._i_a_idx] * self.i_a_limit
        self.i_b=state[self._i_b_idx] * self.i_b_limit
        self.i_c=state[self._i_c_idx] * self.i_c_limit
        self.u_sd=state[self._u_sd_idx] * self.u_sd_limit
        self.u_sq=state[self._u_sq_idx] * self.u_sq_limit
        self.u_a=state[self._u_a_idx] * self.u_a_limit
        self.u_b=state[self._u_b_idx] * self.u_b_limit
        self.u_c=state[self._u_c_idx] * self.u_c_limit
        self.omega=state[self._omega_idx] * self.omega_limit
        self.torque=state[self._torque_idx] * self.torque_limit
        current=np.sqrt(self.i_sd**2 + self.i_sq**2)
        return current



if __name__ == '__main__':

    """
        motor type:     'PMSM'      Permanent Magnet Synchronous Motor
                        'SynRM'     Synchronous Reluctance Motor

        control type:   'SC'         Speed Control
                        'TC'         Torque Control
                        'CC'         Current Control

        action_type:    'Cont'   Continuous Action Space in ABC-Coordinates
                        'Finite'    Discrete Action Space
    """

    motor_type = 'PMSM'
    control_type = 'TC'
    action_type = 'Cont'

    env_id = action_type + '-' + control_type + '-' + motor_type + '-v0'


    # definition of the plotted variables
    #external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_sd', 'i_sq', 'i_abs', 'u_sd', 'u_sq']]
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_a', 'i_b', 'u_a', 'u_b']]

    dashboard = MotorDashboard(additional_plots=external_ref_plots,
                               time_plot_width=20000)

    emrax_208_HV_parameters = {
        'motor_parameter': {
            'p':10,               # number of pole pairs
            'r_s':12e-3,          # stator resistance (ohm)
            'l_d':125e-6,         # d-axis inductance (H)
            'l_q':130e-6,         # q-axis inductance (H)
            'psi_p':39.3e-3,      # magnetic flux of the permanent magnet (Vs)
            'j_rotor':23e-3       # rotor inertia (kg/m^2)
        },
        'nominal_values': {
            'omega':4000*2*np.pi/60,  # angular velocity in rad/s
            'i':141,                  # motor current in amps (peak)
            'u':300                   # nominal voltage in volts (docs say this should be the amplitude, but it
                                      # in fact seems to be the peak-peak value)
        },
        'limit_values': {
            'omega':6000*2*np.pi/60,  # angular velocity in rad/s
            'i':280,                  # motor current in amps (peak)
            'u':450                   # nominal voltage in volts (docs say this should be the amplitude, but it
                                      # in fact seems to be the peak-peak value)
        }
    }

    emrax_208_LV_parameters = {
        'motor_parameter': {
            'p':10,               # number of pole pairs
            'r_s':0.9e-3,         # stator resistance (ohm)
            'l_d':7.2e-6,         # d-axis inductance (H)
            'l_q':7.5e-6,         # q-axis inductance (H)
            'psi_p':9.5e-3,       # magnetic flux of the permanent magnet (Vs)
            'j_rotor':23e-3       # rotor inertia (kg/m^2)
        },
        'nominal_values': {
            'omega':4000*2*np.pi/60,  # angular velocity in rad/s
            'i':565,                  # motor current in amps (peak)
            'u':120                   # nominal voltage in volts (docs say this should be the amplitude, but it
                                      # in fact seems to be the peak-peak value)
        },
        'limit_values': {
            'omega':6000*2*np.pi/60,  # angular velocity in rad/s
            'i':1100,                 # motor current in amps (peak)
            'u':140                   # nominal voltage in volts (docs say this should be the amplitude, but it
                                      # in fact seems to be the peak-peak value)
        }
    }

    emrax_208_LV_43_parameters = {
        'motor_parameter': {
            'p':10,               # number of pole pairs
            'r_s':1e-3,         # stator resistance (ohm)
            'l_d':4.1e-6,         # d-axis inductance (H)
            'l_q':4.3e-6,         # q-axis inductance (H)
            'psi_p':9.5e-3,       # magnetic flux of the permanent magnet (Vs)
            'j_rotor':23e-3       # rotor inertia (kg/m^2)
        },
        'nominal_values': {
            'omega':2000*2*np.pi/60,  # angular velocity in rad/s
            'i':550,                  # motor current in amps (peak)
            'u':120                   # nominal voltage in volts (docs say this should be the amplitude, but it
                                      # in fact seems to be the peak-peak value)
        },
        'limit_values': {
            'omega':4000*2*np.pi/60,  # angular velocity in rad/s
            'i':990,                 # motor current in amps (peak)
            'u':140                   # nominal voltage in volts (docs say this should be the amplitude, but it
                                      # in fact seems to be the peak-peak value)
        }
    }

    battery_parameters = {
        'voltage':83,
        'parameters': {
            'R':27.82e-3,
            'C':1
        }
    }

    propeller_parameters = {
        'load_parameter': {
            'a':1e-3,
            'b':5e-4,
            'c':6e-4,
            'j_load':200e-3#100e-3
        },
        'limits': {
            'omega':6000*2*np.pi/60
        }
    }

    propeller_load = PolynomialStaticLoad(load_parameter=propeller_parameters['load_parameter'], limits=propeller_parameters['limits'])
    constant_load = ConstantSpeedLoad(omega_fixed=3600*np.pi/30)

    converter = ContB6BridgeConverter()

    supply = RCVoltageSupply(battery_parameters['voltage'], battery_parameters['parameters'])
    reference_generator = ConstReferenceGenerator(reference_value=.625)

    wrapper = CurrentVectorProcessor()
    physical_system_wrappers = []
    physical_system_wrappers.append(wrapper)

    # initialize the gym-electric-motor environment
    env = gem.make(env_id, supply=supply, motor=emrax_208_LV_43_parameters,reference_generator=reference_generator,
                   load=propeller_load,
                   converter=converter,
                   #visualization=dashboard,
                   physical_system_wrappers = physical_system_wrappers)

    """
    initialize the controller

    Args:
        environment                     gym-electric-motor environment
        external_ref_plots (optional)   plots of the environment, to plot all reference values
        stages (optional)               structure of the controller
        automated_gain (optional)       if True (default), the controller will be tune automatically
        a (optional)                    tuning parameter of the Symmetrical Optimum (default: 4)

        additionally for TC or SC:
        torque_control (optional)       mode of the torque controller, 'interpolate' (default), 'analytical' or 'online'
        plot_torque(optional)           plot some graphs of the torque controller (default: True)
        plot_modulation (optional)      plot some graphs of the modulation controller (default: False)

    """

    controller = Controller.make(env, torque_control='analytical', plot_torque=False)

    state, reference = env.reset()
    id = []
    iq = []
    ia = []
    ib = []
    ic = []
    ud = []
    uq = []
    ua = []
    ub = []
    uc = []
    omega = []
    torque = []
    isup = []

    # simulate the environment
    for i in range(20001):
        #env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        id.append(wrapper.i_sd)
        iq.append(wrapper.i_sq)
        ia.append(wrapper.i_a)
        ib.append(wrapper.i_b)
        ic.append(wrapper.i_c)
        ud.append(wrapper.u_sd)
        uq.append(wrapper.u_sq)
        ua.append(wrapper.u_a)
        ub.append(wrapper.u_b)
        uc.append(wrapper.u_c)
        omega.append(wrapper.omega)
        torque.append(wrapper.torque)
        isup.append(converter.i_sup([wrapper.i_a, wrapper.i_b, wrapper.i_c]))
        if done:
            print('Done')
            env.reset()
            controller.reset()

    motor_dict = {'id': id, 'iq': iq, 'ud': ud, 'uq': uq, 'ia': ia, 'ib': ib, 'ic': ic, 'isup': isup, 'ua': ua, 'ub': ub, 'uc': uc, 'omega': omega, 'torque': torque}
    motor = pd.DataFrame(data=motor_dict)
    motor['mech_pwr'] = motor['omega'] * motor['torque']
    motor['elec_pwr_dc'] = np.sqrt(motor['id']**2 + motor['iq']**2) * np.sqrt(motor['ud']**2 + motor['uq']**2)
    motor['elec_pwr_ac'] = motor.ia * motor.ua + motor.ib * motor.ub + motor.ic * motor.uc
    motor['eff'] = motor.mech_pwr/motor.elec_pwr_ac
    motor['rpm'] = motor.omega * 30 / np.pi
    motor['v_rms'] = np.sqrt(motor.ua ** 2 + motor.ub ** 2 + motor.uc ** 2)
    motor['ia_amp'] = motor.ia.rolling(1000).max()
    motor['ib_amp'] = motor.ib.rolling(1000).max()
    motor['ic_amp'] = motor.ic.rolling(1000).max()
    motor['ua_amp'] = motor.ua.rolling(1000).max()
    motor['ub_amp'] = motor.ub.rolling(1000).max()
    motor['uc_amp'] = motor.uc.rolling(1000).max()
    motor['ka'] = motor.torque / ((motor.ia_amp + motor.ib_amp + motor.ic_amp) / 3 / np.sqrt(2))
    motor['ke'] = ((motor.ua_amp + motor.ub_amp + motor.uc_amp) / 3 / np.sqrt(0.66667)) / motor.rpm
    motor[['rpm', 'torque']].plot(grid=True, linestyle='none', marker='.', subplots=True)
    motor[['ka', 'ke']].plot(grid = True, linestyle='none', marker='.')
    motor[['ia', 'ib', 'ic', 'isup']].plot(grid=True)
    motor[['id', 'iq']].plot(grid=True, linestyle='none', marker='.')
    motor[['ua', 'ub', 'uc', 'v_rms']].plot(grid=True, linestyle='none', marker='.')
    motor[['ud', 'uq']].plot(grid=True, linestyle='none', marker='.')
    motor[['mech_pwr','elec_pwr_dc', 'elec_pwr_ac']].plot(grid=True, linestyle='none', marker='.')
    plt.figure()
    motor.eff.plot(grid=True, linestyle='none', marker='.')
    plt.show()


#    input("Press Enter to continue...")

    env.close()
