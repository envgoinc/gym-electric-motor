from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import RCVoltageSupply
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
from gym_electric_motor.reference_generators import ConstReferenceGenerator
from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper
import numpy as np

import gym
import numpy as np

class CurrentVectorProcessor(PhysicalSystemWrapper):
    """Adds an ``i_abs`` state to the systems state vector that is the root of the squared sum of the currents ``i_sd`` and `` i_sq``."""

    def __init__(self, physical_system=None):
        self._i_sd_idx = None
        self._i_sq_idx = None
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
        return np.sqrt(state[self._i_sd_idx]**2 + state[self._i_sq_idx]**2)



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
    control_type = 'SC'
    action_type = 'Cont'

    env_id = action_type + '-' + control_type + '-' + motor_type + '-v0'


    # definition of the plotted variables
    #external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_sd', 'i_sq', 'i_abs', 'u_sd', 'u_sq']]
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_a', 'u_a', 'u_b']]

    emrax_208_HV = {
        'motor_parameter': {
            'p':10,               # number of pole pairs
            'r_s':12e-3,          # stator resistance (ohm)
            'l_d':125e-6,         # d-axis inductance (H)
            'l_q':130e-6,         # q-axis inductance (H)
            'psi_p':39.3e-3,      # magnetic flux of the permanent magnet (Vs)
            'j_rotor':23e-3       # rotor inertia (kg/m^2)
        },
        'nominal_values': {
            'omega':3000*2*np.pi/60,  # angular velocity in rad/s
            'i':141,                  # motor current in amps (peak)
            'u':200                   # nominal voltage in volts (peak)
        },
        'limit_values': {
            'omega':6000*2*np.pi/60,  # angular velocity in rad/s
            'i':280,                  # motor current in amps (peak)
            'u':400                   # nominal voltage in volts (peak)
        }
    }

    battery = {
        'voltage':100,
        'parameters': {
            'R':27.82e-3,
            'C':1
        }
    }

    propeller_parameters = {
        'load_parameter': {
            'a':1e-3,
            'b':1e-4,
            'c':5e-4,
            'j_load':100e-3
        },
        'limits': {
            'omega':6000*2*np.pi/60
        }
    }

    propeller_load = PolynomialStaticLoad(load_parameter=propeller_parameters['load_parameter'], limits=propeller_parameters['limits'])

    supply = RCVoltageSupply(battery['voltage'], battery['parameters'])
    reference_generator = ConstReferenceGenerator(reference_value=1)

    physical_system_wrappers = []
    physical_system_wrappers.append(CurrentVectorProcessor())

    # initialize the gym-electric-motor environment
    env = gem.make(env_id, supply=supply, motor=emrax_208_HV,reference_generator=reference_generator,
                   load=propeller_load,
                   visualization=MotorDashboard(additional_plots=external_ref_plots),
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

    controller = Controller.make(env, external_ref_plots=external_ref_plots, torque_control='analytical')

    state, reference = env.reset()

    # simulate the environment
    for i in range(20001):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        if done:
            print('Done')
            env.reset()
            controller.reset()

    print(type(state))
    input("Press Enter to continue...")

    env.close()
