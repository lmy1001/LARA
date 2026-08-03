"""Small NumPy-aware ZeroMQ transport used by the evaluation examples."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable

import msgpack
import numpy as np
import zmq


class MsgpackSerializer:
    """Serialize nested dictionaries without requiring PyTorch on the client."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(
            data,
            default=MsgpackSerializer._encode,
            use_bin_type=True,
        )

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(
            data,
            object_hook=MsgpackSerializer._decode,
            raw=False,
        )

    @staticmethod
    def _encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            return {"__ndarray__": buffer.getvalue()}
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize value of type {type(value).__name__}")

    @staticmethod
    def _decode(value: dict[str, Any]) -> Any:
        if "__ndarray__" in value:
            return np.load(io.BytesIO(value["__ndarray__"]), allow_pickle=False)
        return value


@dataclass(frozen=True)
class EndpointHandler:
    handler: Callable[..., Any]
    requires_input: bool = True


class BaseInferenceServer:
    """Serve named request/reply endpoints over ZeroMQ."""

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def register_endpoint(
        self,
        name: str,
        handler: Callable[..., Any],
        requires_input: bool = True,
    ) -> None:
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict[str, str]:
        return {"status": "ok"}

    def _kill_server(self) -> dict[str, str]:
        self.running = False
        return {"status": "stopping"}

    def run(self) -> None:
        address = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Inference server listening on {address}", flush=True)
        try:
            while self.running:
                request = MsgpackSerializer.from_bytes(self.socket.recv())
                try:
                    endpoint = request.get("endpoint", "get_action")
                    if endpoint not in self._endpoints:
                        raise ValueError(f"Unknown endpoint: {endpoint}")
                    endpoint_handler = self._endpoints[endpoint]
                    response = (
                        endpoint_handler.handler(request.get("data", {}))
                        if endpoint_handler.requires_input
                        else endpoint_handler.handler()
                    )
                except Exception as error:
                    response = {"error": f"{type(error).__name__}: {error}"}
                self.socket.send(MsgpackSerializer.to_bytes(response))
        except KeyboardInterrupt:
            print("Inference server stopped", flush=True)
        finally:
            self.close()

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class BaseInferenceClient:
    """Request/reply client with deterministic timeouts and socket recovery."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 120_000,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._init_socket()

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> Any:
        if self.socket is None:
            raise RuntimeError("Inference client is closed")
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(MsgpackSerializer.to_bytes(request))
            response = MsgpackSerializer.from_bytes(self.socket.recv())
        except zmq.Again as error:
            self._init_socket()
            raise TimeoutError(f"Inference request to {self.host}:{self.port} timed out") from error
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Inference server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            return self.call_endpoint("ping") == {"status": "ok"}
        except (RuntimeError, TimeoutError, zmq.ZMQError):
            return False

    def kill_server(self) -> None:
        self.call_endpoint("kill")

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

    def __enter__(self) -> BaseInferenceClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """Client endpoint used by simulator processes."""

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        return self.call_endpoint("get_action", observations)


class RobotInferenceServer(BaseInferenceServer):
    """Expose a policy's ``get_action`` method."""

    def __init__(self, policy: Any, host: str = "*", port: int = 5555):
        super().__init__(host=host, port=port)
        self.register_endpoint("get_action", policy.get_action)
