import os
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

import logging
logger = logging.getLogger(__name__)

class EntropyCalibrator2:
    """
    Dummy calibrator fallback if TensorRT is not available.
    """
    pass

if trt is not None:
    class EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
        """
        INT8 Calibration dataset loader for TensorRT.
        Uses EntropyCalibrator2 which is recommended for CNNs and Transformers.
        """
        def __init__(self, data_loader, cache_file="calibration.cache", batch_size=8, input_name="input"):
            # Whenever you specify a custom constructor for a TensorRT class,
            # you MUST call the constructor of the parent explicitly.
            super(EntropyCalibrator2, self).__init__()
            
            self.data_loader = data_loader # Generator yielding numpy arrays
            self.cache_file = cache_file
            self.batch_size = batch_size
            self.input_name = input_name
            self.current_index = 0
            
            # Get first batch to determine shapes
            try:
                self.first_batch = next(self.data_loader)
                # Allocate device memory for this batch
                self.device_input = cuda.mem_alloc(self.first_batch.nbytes)
            except StopIteration:
                logger.error("Calibration data loader is empty!")
                raise

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            """
            Return a list of device pointers to the calibration batch.
            """
            try:
                # If we just started, use the first batch we peeked at
                if self.current_index == 0:
                    batch = self.first_batch
                else:
                    batch = next(self.data_loader)
                
                # Copy to device
                cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
                self.current_index += 1
                return [int(self.device_input)]
            except StopIteration:
                # When we're out of batches, we return None
                return None

        def read_calibration_cache(self):
            # If there is a cache, use it instead of calibrating again
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    logger.info(f"Using calibration cache to save time: {self.cache_file}")
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                logger.info(f"Saving calibration cache for future builds: {self.cache_file}")
                f.write(cache)

        def free(self):
            if self.device_input:
                self.device_input.free()
