import numpy as np
import logging

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

logger = logging.getLogger(__name__)

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTInferenceEngine:
    def __init__(self, engine_file_path):
        """
        Loads the TensorRT engine and initializes I/O buffers.
        """
        if trt is None:
            raise RuntimeError("TensorRT is not installed.")
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_file_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        self.input_names = []
        self.output_names = []
        
        # We delay buffer allocation until the first forward pass, 
        # or we could do it for a default batch size if dynamic.
        self._buffers_allocated = False

    def allocate_buffers(self, input_shapes):
        """
        Allocates pinned host memory and device memory based on actual runtime shapes.
        """
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_names = []
        self.output_names = []
        
        # Free old buffers if re-allocating
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            
            if is_input:
                self.input_names.append(name)
                # Set dynamic shape
                self.context.set_binding_shape(i, input_shapes[name])
                shape = self.context.get_binding_shape(i)
            else:
                self.output_names.append(name)
                shape = self.context.get_binding_shape(i)
                
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append device pointer to bindings list
            self.bindings.append(int(device_mem))
            
            if is_input:
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                
        self._buffers_allocated = True

    def __call__(self, **kwargs):
        """
        Runs async inference on the loaded TensorRT engine.
        kwargs: mapping of input_name -> numpy array
        """
        # Determine shapes
        shapes = {k: v.shape for k, v in kwargs.items()}
        
        # If not allocated or shape changed, re-allocate
        if not self._buffers_allocated:
            self.allocate_buffers(shapes)
        else:
            # Check if shapes changed
            for i, name in enumerate(self.input_names):
                if shapes[name] != self.context.get_binding_shape(i):
                    self.allocate_buffers(shapes)
                    break

        # Copy inputs to host buffers
        for i, name in enumerate(self.input_names):
            np.copyto(self.inputs[i].host, kwargs[name].ravel())

        # Transfer input data to device asynchronously
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            
        # Execute model asynchronously
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from device asynchronously
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            
        # Synchronize the stream
        self.stream.synchronize()
        
        # Return a dictionary of outputs
        results = {}
        for i, name in enumerate(self.output_names):
            out_shape = self.context.get_binding_shape(self.input_names.index(name) if name in self.input_names else len(self.input_names) + i)
            # Actually get binding index for output
            bind_idx = self.engine.get_binding_index(name)
            out_shape = self.context.get_binding_shape(bind_idx)
            results[name] = self.outputs[i].host.reshape(out_shape).copy()
            
        return results
