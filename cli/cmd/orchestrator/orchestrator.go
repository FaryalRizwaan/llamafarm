package orchestrator

import (
	"errors"
	"fmt"
	"os"
)

// ErrServiceAlreadyRunning indicates that a service is already running
var ErrServiceAlreadyRunning = errors.New("service is already running")

// NativeOrchestrator manages the infrastructure layer for running services.
//
// Responsibilities:
// - Binary download and management (PyApp binaries)
// - Process management (start/stop/tracking)
// - Environment variable builders for services
//
// NOT responsible for:
// - Service lifecycle management (see ServiceManager)
// - Dependency resolution (see ServiceManager)
// - Health checking (see ServiceManager)
//
// Use ServiceManager for high-level service orchestration.
// NativeOrchestrator provides the infrastructure that ServiceManager builds on.
type NativeOrchestrator struct {
	binaryMgr  *BinaryManager
	processMgr *ProcessManager
	serverURL  string // current runtime URL (may be adjusted for port conflicts)
}

// NewOrchestrator creates a new orchestrator that launches PyApp binaries.
func NewOrchestrator(serverURL string) (*NativeOrchestrator, error) {
	procMgr, err := NewProcessManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create process manager: %w", err)
	}

	// Create binary manager with version from LF_VERSION env var or CLI version
	version := os.Getenv("LF_VERSION")
	binaryMgr, err := NewBinaryManager(version)
	if err != nil {
		return nil, fmt.Errorf("failed to create binary manager: %w", err)
	}

	orchestrator := &NativeOrchestrator{
		processMgr: procMgr,
		binaryMgr:  binaryMgr,
		serverURL:  serverURL,
	}

	// Download binaries if needed
	if err := binaryMgr.EnsureBinaries(); err != nil {
		return nil, fmt.Errorf("failed to ensure binaries: %w", err)
	}

	return orchestrator, nil
}

// getBinaryEnv builds environment variables for a service process.
func (no *NativeOrchestrator) getBinaryEnv(envKeysWithDefaults map[string]string) []string {
	var env []string

	// Inherit core environment keys (including Windows home directory vars
	// needed by Python's Path.home() and getpass.getuser() inside PyApp binaries)
	for _, key := range []string{
		"HOME", "USER", "USERNAME", "LOGNAME", "TMPDIR", "TEMP", "TMP",
		"LF_DATA_DIR", "PATH", "SYSTEMROOT",
		"USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
	} {
		if val := os.Getenv(key); val != "" {
			env = append(env, fmt.Sprintf("%s=%s", key, val))
		}
	}

	// Add service-specific environment variables
	for key, val := range envKeysWithDefaults {
		if val != "" {
			expandedVal := os.ExpandEnv(val)
			env = append(env, fmt.Sprintf("%s=%s", key, expandedVal))
		} else if envVal := os.Getenv(key); envVal != "" {
			// Empty default means "inherit from parent environment if set"
			env = append(env, fmt.Sprintf("%s=%s", key, envVal))
		}
	}

	return env
}

// StopAllProcesses stops all native processes
func (no *NativeOrchestrator) StopAllProcesses() {
	if no.processMgr != nil {
		no.processMgr.StopAllProcesses()
	}
}

// GetProcessManager returns the process manager
func (no *NativeOrchestrator) GetProcessManager() *ProcessManager {
	return no.processMgr
}
