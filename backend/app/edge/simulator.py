import os
import sys
import logging

logger = logging.getLogger(__name__)

def simulate_jetson_constraints():
    """
    Simulates the resource constraints of an NVIDIA Jetson Nano (4GB RAM, 4 CPU Cores).
    This function should be called at the very beginning of the application entry point.
    """
    logger.info("Initializing Jetson Nano constraint simulator...")

    # 1. Throttle CPU threads to simulate the 4-core ARM Cortex-A57
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    logger.info("CPU threads throttled to 4 cores.")

    # 2. Cap RAM to 4GB using the `resource` module (Unix only)
    try:
        import resource
        
        # 4GB in bytes
        MAX_RAM_BYTES = 4 * 1024 * 1024 * 1024 
        
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        
        # Set new limit (taking the min of hard limit and 4GB)
        new_limit = min(hard, MAX_RAM_BYTES)
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        
        logger.info(f"RAM successfully capped to 4GB (RLIMIT_AS={new_limit}).")
        
    except ImportError:
        logger.warning("The 'resource' module is not available on this OS (Windows). RAM capping is skipped.")
    except Exception as e:
        logger.error(f"Failed to set memory limits: {e}")

if __name__ == "__main__":
    simulate_jetson_constraints()
