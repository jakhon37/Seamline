"""
GPU Memory Tracking Utilities
"""
from typing import Dict, Optional

import torch


class MemoryTracker:
    """Track GPU memory usage during training."""

    def __init__(self, enabled: bool = True, device_id: int = 0):
        """
        Args:
            enabled: Whether to track memory
            device_id: GPU device ID to track
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.peak_memory_mb = 0

        if self.enabled:
            # Validate and set device
            device_count = torch.cuda.device_count()
            if device_count == 0:
                self.enabled = False
                self.device_id = 0
            else:
                self.device_id = min(device_id, device_count - 1)
                try:
                    torch.cuda.set_device(self.device_id)
                    torch.cuda.reset_peak_memory_stats(self.device_id)
                except Exception:
                    # If device init fails, disable tracking
                    self.enabled = False
        else:
            self.device_id = 0

    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """
        Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats in MB, or None if not available
        """
        if not self.enabled:
            return None

        allocated = torch.cuda.memory_allocated(self.device_id) / 1e6
        reserved = torch.cuda.memory_reserved(self.device_id) / 1e6
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1e6

        # Update peak
        self.peak_memory_mb = max(self.peak_memory_mb, max_allocated)

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
            "peak_mb": self.peak_memory_mb,
        }

    def get_formatted_stats(self) -> str:
        """
        Get formatted memory statistics string.

        Returns:
            Formatted string with memory usage
        """
        stats = self.get_memory_stats()
        if stats is None:
            return "GPU memory tracking not available"

        return (
            f"GPU Memory - Allocated: {stats['allocated_mb']:.1f}MB, "
            f"Reserved: {stats['reserved_mb']:.1f}MB, "
            f"Peak: {stats['peak_mb']:.1f}MB"
        )

    def reset_peak(self):
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device_id)
            self.peak_memory_mb = 0

    @staticmethod
    def get_gpu_memory_summary() -> str:
        """
        Get summary of GPU memory for all devices.

        Returns:
            Formatted summary string
        """
        if not torch.cuda.is_available():
            return "CUDA not available"

        lines = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e6
            total = torch.cuda.get_device_properties(i).total_memory / 1e6
            lines.append(
                f"GPU {i}: {allocated:.1f}MB / {total:.1f}MB " f"({allocated/total*100:.1f}% used)"
            )

        return "\n".join(lines)
