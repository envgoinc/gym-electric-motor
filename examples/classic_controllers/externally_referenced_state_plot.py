from gym_electric_motor.visualization.motor_dashboard_plots import StatePlot


class ExternallyReferencedStatePlot(StatePlot):
    """Plot that displays environments states together with externally generated references.
            These could be for example references that are generated intermediately within a cascaded controller.
            Usage Example
            -------------
            .. code-block:: python
                :emphasize-lines: 1,12
                my_externally_referenced_plot = ExternallyReferencedStatePlot(state='i_sd')
                env = gem.make(
                    'DqCont-SC-PMSM-v0',
                    visualization=dict(additional_plots=(my_externally_referenced_plot,),
                )
                done = True
                for _ in range(10000):
                    if done:
                        state, reference = env.reset()
                    external_reference_value = my_external_isd_reference_generator.get_reference()
                    my_externally_referenced_plot.external_reference(external_reference_value)
                    env.render()
                    action = env.action_space.sample()
                    (state, reference), reward, done, _ = env.step(action)
    """

    def __init__(self, state):
        super().__init__(state)
        self.state_plot = state

    def set_reference(self, ref_states):
        self._referenced = self.state_plot in ref_states

    def set_env(self, env):
        super().set_env(env)

    def external_reference(self, value):
        self._ref_data[self.data_idx] = value * self._limits
