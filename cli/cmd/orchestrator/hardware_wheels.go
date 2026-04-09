package orchestrator

// HardwarePackageSpec defines a Python package that requires hardware-specific installation
type HardwarePackageSpec struct {
	Name              string                        // Package name (e.g., "torch")
	Version           string                        // Version constraint (e.g., ">=2.0.0")
	UseIndexURL       bool                          // If true, use --index-url; if false, use --extra-index-url
	WheelURLs         map[HardwareCapability]string // Hardware-specific wheel index URLs
	FallbackToDefault bool                          // If true and URL is empty, use default PyPI
}

// PyTorchSpec defines the hardware-specific installation for PyTorch
var PyTorchSpec = HardwarePackageSpec{
	Name:              "torch",
	Version:           ">=2.0.0",
	UseIndexURL:       true, // PyTorch uses --index-url for hardware-specific builds
	FallbackToDefault: true,
	WheelURLs: map[HardwareCapability]string{
		HardwareCPU:   "https://download.pytorch.org/whl/cpu",
		HardwareCUDA:  "", // Empty = use default PyPI (has CUDA builds)
		HardwareMetal: "", // Empty = use default PyPI (has Metal support)
		HardwareROCm:  "https://download.pytorch.org/whl/rocm6.4",
	},
}

// GetComponentPackages returns the hardware-dependent packages for a given component.
// It looks up the component in ServiceGraph and returns its HardwarePackages field.
func GetComponentPackages(componentName string) []HardwarePackageSpec {
	if svc, exists := ServiceGraph[componentName]; exists {
		return svc.HardwarePackages
	}
	return []HardwarePackageSpec{}
}
