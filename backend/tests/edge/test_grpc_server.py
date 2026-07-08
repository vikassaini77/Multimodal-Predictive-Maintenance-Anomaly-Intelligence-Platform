import pytest
import asyncio
import grpc
import numpy as np
import os

try:
    from backend.app.edge import edge_pb2
    from backend.app.edge import edge_pb2_grpc
    from backend.app.edge.server import serve
except ImportError:
    edge_pb2 = None
    edge_pb2_grpc = None

# We use pytest-asyncio to run async tests
@pytest.mark.asyncio
@pytest.mark.skipif(edge_pb2 is None, reason="gRPC modules not generated")
async def test_concurrent_edge_inferences():
    """
    Test the EdgeInferenceService under high concurrency (100 concurrent requests).
    Verifies that the TensorRT engine execution context and buffers don't encounter race conditions.
    """
    # 1. Start the server in the background
    server_task = asyncio.create_task(serve())
    
    # Wait for server to spin up
    await asyncio.sleep(1)
    
    try:
        # 2. Setup client channel
        channel = grpc.aio.insecure_channel('localhost:50051')
        stub = edge_pb2_grpc.EdgeInferenceStub(channel)
        
        # 3. Create dummy requests
        def create_req():
            sensor_bytes = np.random.randn(1, 2, 50).astype(np.float32).tobytes()
            img_bytes = b'\\x00' * 512
            return edge_pb2.InferenceRequest(
                sensor_data=sensor_bytes,
                image_data=img_bytes
            )

        # 4. Fire 100 concurrent requests
        requests = [create_req() for _ in range(100)]
        
        async def call_predict(req):
            response = await stub.PredictAnomaly(req)
            assert response.risk_score >= 0.0
            assert response.explanation is not None
            return response
            
        # Run them all concurrently
        results = await asyncio.gather(*(call_predict(r) for r in requests), return_exceptions=True)
        
        # 5. Assert all succeeded
        for res in results:
            assert not isinstance(res, Exception), f"A request failed with exception: {res}"
            
        print(f"Successfully processed {len(results)} concurrent requests.")
        
        await channel.close()

    finally:
        # 6. Tear down server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
