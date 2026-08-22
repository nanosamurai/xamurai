python -m grpc_tools.protoc -I ../proto --python_out=. --grpc_python_out=. \
  ../proto/stream.proto
sed -i 's/^import stream_pb2/from . import stream_pb2/' stream_pb2_grpc.py
