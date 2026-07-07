import os
import logging

try:
    import tensorrt as trt
except ImportError:
    trt = None

logger = logging.getLogger(__name__)

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True, int8_mode=False, calibrator=None):
    """
    Builds a TensorRT engine from an ONNX file.
    """
    if trt is None:
        logger.error("TensorRT is not installed. Cannot build engine.")
        return False
        
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Initialize Builder
    builder = trt.Builder(TRT_LOGGER)
    
    # Use EXPLICIT_BATCH flag for dynamic shapes or standard ONNX
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Initialize Parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    if not os.path.exists(onnx_file_path):
        logger.error(f"ONNX file {onnx_file_path} not found.")
        return False
        
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return False
            
    # Configure Builder
    config = builder.create_builder_config()
    
    # Workspace size (1GB for memory optimization passes)
    # config.max_workspace_size is deprecated in TRT 8.x, set_memory_pool_limit is used
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    else:
        config.max_workspace_size = 1 << 30
        
    # Set Optimization Profile for dynamic axes
    profile = builder.create_optimization_profile()
    
    # We need to know the inputs to set dynamic shapes
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape
        
        # Example: if shape has -1, it means dynamic. 
        # For simplicity, we just set a reasonable range based on our models
        if -1 in shape:
            min_shape = []
            opt_shape = []
            max_shape = []
            
            for dim in shape:
                if dim == -1:
                    min_shape.append(1)
                    opt_shape.append(4) # typical batch
                    max_shape.append(32) # max batch
                else:
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)
            
            # Note: For SensorTower seq_len, we might need different dims, 
            # assuming seq_len doesn't vary dramatically or is handled here.
            # E.g., batch_size, 2, seq_len where seq_len might be 50.
            # If shape is (-1, 2, -1):
            if name == "sensor_input":
                min_shape = (1, 2, 50)
                opt_shape = (4, 2, 50)
                max_shape = (32, 2, 100)
            elif name == "visual_input":
                min_shape = (1, 3, 224, 224)
                opt_shape = (4, 3, 224, 224)
                max_shape = (32, 3, 224, 224)
                
            profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            
    config.add_optimization_profile(profile)

    # FP16 Mode
    if fp16_mode and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)
        
    # INT8 Mode
    if int8_mode and builder.platform_has_fast_int8:
        logger.info("Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is not None:
            config.int8_calibrator = calibrator
        else:
            logger.warning("INT8 mode requested but no calibrator provided!")

    # Build the engine
    logger.info("Building TensorRT engine... (this can take a few minutes)")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        logger.error("Failed to build engine.")
        return False
        
    # Save the engine
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
        
    logger.info(f"Engine successfully saved to {engine_file_path}")
    return True

if __name__ == "__main__":
    # Example usage for testing
    build_engine(
        "backend/app/deployment/onnx_models/sensor_tower_opt.onnx", 
        "backend/app/deployment/onnx_models/sensor_tower.plan",
        fp16_mode=True
    )
