"""
Training Profiler Utilities
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from typing import Optional
import os


class TrainingProfiler:
    """Profiler for identifying training bottlenecks."""
    
    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "./profiler_output",
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        repeat: int = 1
    ):
        """
        Args:
            enabled: Whether to enable profiling
            output_dir: Directory to save profiler outputs
            wait_steps: Number of steps to skip at the start
            warmup_steps: Number of warmup steps
            active_steps: Number of steps to profile
            repeat: Number of times to repeat the cycle
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.prof = None
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat
        
        if self.enabled:
            os.makedirs(output_dir, exist_ok=True)
    
    def start(self, tensorboard_trace: bool = True):
        """
        Start profiling.
        
        Args:
            tensorboard_trace: Whether to export TensorBoard trace
        """
        if not self.enabled:
            return
        
        # Create profiler schedule
        prof_schedule = schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat
        )
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        self.prof = profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=self._trace_handler if tensorboard_trace else None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.prof.__enter__()
    
    def step(self):
        """Step the profiler (call after each training iteration)."""
        if self.enabled and self.prof is not None:
            self.prof.step()
    
    def stop(self):
        """Stop profiling and save results."""
        if not self.enabled or self.prof is None:
            return
        
        self.prof.__exit__(None, None, None)
        
        # Print summary
        print("\n" + "="*80)
        print("PROFILER SUMMARY - Top operations by CUDA time:")
        print("="*80)
        print(self.prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=20
        ))
        print("="*80)
        
        # Save detailed trace
        trace_file = os.path.join(self.output_dir, "profiler_trace.json")
        self.prof.export_chrome_trace(trace_file)
        print(f"\nDetailed trace saved to: {trace_file}")
        print("View in Chrome: chrome://tracing")
        
        self.prof = None
    
    def _trace_handler(self, prof):
        """Handler for trace ready events."""
        # Export to TensorBoard
        output_path = os.path.join(self.output_dir, f"profiler_step_{prof.step_num}")
        prof.export_chrome_trace(output_path + ".json")
        
        # Print top operations
        if prof.step_num == self.wait_steps + self.warmup_steps + self.active_steps:
            print(f"\n[Profiler] Step {prof.step_num} summary:")
            print(prof.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=10
            ))
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def profile_section(name: str, enabled: bool = True):
    """
    Context manager for profiling a specific section of code.
    
    Args:
        name: Name of the section
        enabled: Whether profiling is enabled
    
    Example:
        with profile_section("data_loading"):
            data = next(dataloader)
    """
    if enabled:
        return record_function(name)
    else:
        return _DummyContext()


class _DummyContext:
    """Dummy context manager when profiling is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
