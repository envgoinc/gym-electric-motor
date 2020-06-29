from examples.agents.simple_controllers import Controller
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard

if __name__ == '__main__':
    env = gem.make(
        'DcPermExDisc-v1',  # replace with 'DcPermExCont-v1' for continuous mode
        # Pass an instance
        visualization=MotorDashboard(plotted_variables=['omega', 'torque', 'i', 'u', 'u_sup'], visu_period=1),
        # motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),
        # Take standard class and pass parameters (Load)
        # load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.06),
        # Pass a string (with extra parameters)
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        # Pass a Class with extra parameters
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator, rg.WienerProcessReferenceGenerator(), rg.StepReferenceGenerator()
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )
    )
    controller = Controller.make('three_point', env)   # works for 'on_off','pi_controller'(cont),'p_controller'(cont), 'cascaded_pi'(cont) too
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in range(100000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
        cum_rew += reward
    print(cum_rew)