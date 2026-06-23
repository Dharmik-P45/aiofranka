"""
Shared memory layout and IPC protocol for aiofranka server/client communication.

State is shared via multiprocessing.shared_memory for zero-copy reads at 1kHz.
Commands are sent via ZMQ REQ/REP with msgpack serialization.
"""

import struct
import time
import numpy as np
from multiprocessing.shared_memory import SharedMemory

# --- Shared Memory Layout ---
# All fields are float64 (8 bytes each) unless noted.
# A uint64 sequence counter is written LAST by server and read FIRST by client
# to detect torn reads.

_FIELDS = [
    # (name, shape, dtype)
    # Robot state (from RobotInterface.state)
    ("qpos", (7,), np.float64),
    ("qvel", (7,), np.float64),
    ("ee", (4, 4), np.float64),
    ("jac", (6, 7), np.float64),
    ("mm", (7, 7), np.float64),
    ("last_torque", (7,), np.float64),
    ("tau_ext_hat_filtered", (7,), np.float64),
    # Controller state
    ("q_desired", (7,), np.float64),
    ("ee_desired", (4, 4), np.float64),
    ("torque", (7,), np.float64),
    ("initial_qpos", (7,), np.float64),
    ("initial_ee", (4, 4), np.float64),
    # Meta
    ("timestamp", (1,), np.float64),
]

# Status flags
STATUS_STOPPED = 0
STATUS_RUNNING = 1
STATUS_ERROR = 2

# Controller type encoding
CTRL_IMPEDANCE = 0
CTRL_PID = 1
CTRL_OSC = 2
CTRL_TORQUE = 3

CTRL_TYPE_MAP = {
    "impedance": CTRL_IMPEDANCE,
    "pid": CTRL_PID,
    "osc": CTRL_OSC,
    "torque": CTRL_TORQUE,
}
CTRL_TYPE_RMAP = {v: k for k, v in CTRL_TYPE_MAP.items()}

# Compute offsets
# Layout: [seq_counter(8)] [fields...] [status(1)] [ctrl_type(1)]
_SEQ_OFFSET = 0
_SEQ_SIZE = 8  # uint64

_field_offsets = {}
_offset = _SEQ_SIZE
for name, shape, dtype in _FIELDS:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    _field_offsets[name] = (_offset, shape, dtype, nbytes)
    _offset += nbytes

_STATUS_OFFSET = _offset
_offset += 1  # uint8
_CTRL_TYPE_OFFSET = _offset
_offset += 1  # uint8

# Error message area (256 bytes for error string)
_ERROR_MSG_OFFSET = _offset
_ERROR_MSG_SIZE = 256
_offset += _ERROR_MSG_SIZE

# Jitter stats area (server writes, client reads)
# Layout: [max_dt_ms: float64] [warn_count: uint64] [error_count: uint64]
_JITTER_OFFSET = _offset
_JITTER_SIZE = 8 + 8 + 8  # 24 bytes
_offset += _JITTER_SIZE

SHM_SIZE = _offset  # Total shared memory size


def shm_name_for_ip(robot_ip: str) -> str:
    return f"aiofranka_{robot_ip.replace('.', '_')}"


def zmq_endpoint_for_ip(robot_ip: str) -> str:
    return f"ipc:///tmp/aiofranka_{robot_ip.replace('.', '_')}.sock"


def progress_file_for_ip(robot_ip: str) -> str:
    return f"/tmp/aiofranka_{robot_ip.replace('.', '_')}.progress"


def pid_file_for_ip(robot_ip: str) -> str:
    return f"/tmp/aiofranka_{robot_ip.replace('.', '_')}.pid"


class StateBlock:
    """Wraps a SharedMemory segment with typed numpy views for robot state."""

    def __init__(self, robot_ip: str, create: bool = False, track: bool = True):
        self.name = shm_name_for_ip(robot_ip)
        if create:
            # Try to clean up stale shm first
            try:
                old = SharedMemory(name=self.name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = SharedMemory(name=self.name, create=True, size=SHM_SIZE)
            # Zero-initialize
            self.shm.buf[:SHM_SIZE] = b'\x00' * SHM_SIZE
        else:
            self.shm = SharedMemory(name=self.name, create=False)
            if not track:
                # Prevent resource_tracker warning for client-side reads
                from multiprocessing import resource_tracker
                resource_tracker.unregister(
                    f"/{self.name}", "shared_memory"
                )
        self.buf = self.shm.buf

    def write_state(self, state: dict):
        """Server-side: write robot state to shared memory. Called at 1kHz."""
        buf = self.buf
        for name, (offset, shape, dtype, nbytes) in _field_offsets.items():
            if name in state:
                arr = np.asarray(state[name], dtype=dtype).ravel()
                buf[offset:offset + nbytes] = arr.tobytes()

        buf[_STATUS_OFFSET] = STATUS_RUNNING

        # Write timestamp
        ts_offset, _, _, ts_nbytes = _field_offsets["timestamp"]
        buf[ts_offset:ts_offset + ts_nbytes] = struct.pack('d', time.time())

        # Increment sequence counter LAST (memory fence for readers)
        seq = struct.unpack('Q', bytes(buf[_SEQ_OFFSET:_SEQ_OFFSET + _SEQ_SIZE]))[0]
        struct.pack_into('Q', buf, _SEQ_OFFSET, seq + 1)

    def read_state(self) -> dict:
        """Client-side: read robot state from shared memory. Retries on torn reads."""
        buf = self.buf
        for _ in range(3):  # retry up to 3 times on torn read
            seq1 = struct.unpack('Q', bytes(buf[_SEQ_OFFSET:_SEQ_OFFSET + _SEQ_SIZE]))[0]

            result = {}
            for name, (offset, shape, dtype, nbytes) in _field_offsets.items():
                raw = bytes(buf[offset:offset + nbytes])
                result[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

            seq2 = struct.unpack('Q', bytes(buf[_SEQ_OFFSET:_SEQ_OFFSET + _SEQ_SIZE]))[0]

            if seq1 == seq2:
                # Clean up: extract scalar timestamp
                result["timestamp"] = float(result["timestamp"][0])
                return result

        # If we still get torn reads, return last attempt anyway
        result["timestamp"] = float(result["timestamp"][0])
        return result

    def write_ctrl_type(self, ctrl_type: str):
        self.buf[_CTRL_TYPE_OFFSET] = CTRL_TYPE_MAP.get(ctrl_type, 0)

    def read_ctrl_type(self) -> str:
        return CTRL_TYPE_RMAP.get(self.buf[_CTRL_TYPE_OFFSET], "impedance")

    def write_status(self, status: int):
        self.buf[_STATUS_OFFSET] = status

    def read_status(self) -> int:
        return self.buf[_STATUS_OFFSET]

    def write_error(self, msg: str):
        self.buf[_STATUS_OFFSET] = STATUS_ERROR
        encoded = msg.encode('utf-8')[:_ERROR_MSG_SIZE - 1] + b'\x00'
        self.buf[_ERROR_MSG_OFFSET:_ERROR_MSG_OFFSET + len(encoded)] = encoded

    def read_error(self) -> str:
        raw = bytes(self.buf[_ERROR_MSG_OFFSET:_ERROR_MSG_OFFSET + _ERROR_MSG_SIZE])
        return raw.split(b'\x00', 1)[0].decode('utf-8', errors='replace')

    def write_jitter_stats(self, max_dt_ms: float, warn_count: int, error_count: int):
        """Server-side: write jitter statistics to shared memory."""
        struct.pack_into('d', self.buf, _JITTER_OFFSET, max_dt_ms)
        struct.pack_into('Q', self.buf, _JITTER_OFFSET + 8, warn_count)
        struct.pack_into('Q', self.buf, _JITTER_OFFSET + 16, error_count)

    def read_jitter_stats(self) -> tuple:
        """Client-side: read jitter statistics. Returns (max_dt_ms, warn_count, error_count)."""
        max_dt_ms = struct.unpack_from('d', self.buf, _JITTER_OFFSET)[0]
        warn_count = struct.unpack_from('Q', self.buf, _JITTER_OFFSET + 8)[0]
        error_count = struct.unpack_from('Q', self.buf, _JITTER_OFFSET + 16)[0]
        return max_dt_ms, warn_count, error_count

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()