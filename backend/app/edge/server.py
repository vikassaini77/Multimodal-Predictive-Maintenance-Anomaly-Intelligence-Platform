import asyncio
import logging
import grpc
from grpc_reflection.v1alpha import reflection

try:
    import backend.app.edge.edge_pb2 as edge_pb2
    import backend.app.edge.edge_pb2_grpc as edge_pb2_grpc
    from backend.app.edge.service import EdgeInferenceService
except ImportError:
    import edge_pb2
    import edge_pb2_grpc
    from service import EdgeInferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def serve():
    # Instantiate the server
    server = grpc.aio.server()
    
    # Add the EdgeInferenceService to the server
    edge_pb2_grpc.add_EdgeInferenceServicer_to_server(
        EdgeInferenceService(), server
    )

    # Enable Server Reflection for tools like grpcurl
    SERVICE_NAMES = (
        edge_pb2.DESCRIPTOR.services_by_name['EdgeInference'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Listen on port 50051 (Standard gRPC port)
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    
    logger.info(f"Edge Inference gRPC Server starting on port {port}...")
    await server.start()
    
    # Wait for termination
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
