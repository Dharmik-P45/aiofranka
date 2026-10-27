"""
Synchronous remote controller client for aiofranka.

Communicates with the aiofranka server process via shared memory (state reads)
and ZMQ (commands). Drop-in replacement for FrankaController with a sync API.
"""

import atexit
import contextlib
import json
import logging
import os
import time
import threading

import msgpack
import numpy as np
import zmq

from aiofranka.ipc import (
    StateBlock,
    zmq_endpoint_for_ip,
    zmq_tcp_connect_endpoint,
    DEFAULT_TCP_PORT,
    STATUS_ERROR,
)

logger = logging.getLogger("aiofranka.remote")

DEFAULT_IP = "172.16.0.2"

# Attributes that get sent to server via ZMQ when assigned
_REMOTE_ATTRS = {
    "kp", "kd", "ki", "ee_kp", "ee_kd", "null_kp", "null_kd",
    "torque_diff_limit", "torque_limit", "clip",
    "q_desired", "ee_desired", "torque",
}

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RST = "\033[0m"


class ServerDiedError(RuntimeError):
    """Raised when the server process dies (e.g. reflex error).

    Attributes:
        server_error: The error message from the server (e.g. libfranka reflex details).
    """
    def __init__(self, server_error: str):
        self.server_error = server_error
        # Build a user-friendly message
        lines = [
            f"\n  {_RED}Server died:{_RST} {server_error}\n",
        ]
        if "reflex" in server_error.lower():
            lines.append(f"  The robot entered {_BOLD}Reflex{_RST} mode (safety stop).")
            lines.append(f"  This usually means the robot hit a joint/velocity/torque limit.\n")
        lines.append(f"  To recover:")
        lines.append(f"    1. Run {_BOLD}aiofranka gravcomp{_RST} to freely move the robot")
        lines.append(f"       to a safe configuration, then Ctrl+C and restart your script.")
        lines.append(f"    2. Or just restart your script (it will auto-recover if possible).\n")
        super().__init__("\n".join(lines))


class FrankaRemoteController:
    """
    Synchronous remote controller for Franka robots.

    Connects to a running aiofranka server process and provides the same
    API as FrankaController, but fully synchronous (no async/await needed).

    The 1kHz control loop runs in the server process. By default (``host=
    None``) this client spawns and owns a server subprocess on the local
    machine, sends commands via ZMQ, and reads state from shared memory
    (zero-copy) -- the original same-machine Server Mode.

    Passing ``host`` instead attaches to a server *already running on a
    different machine* (started there with ``aiofranka start-server``, e.g.
    as a systemd service). No subprocess is spawned and nothing is torn down
    on ``stop()`` -- this client is just one of possibly several consumers
    of a long-lived remote control loop. State reads go over the network
    (via a ``get_state`` ZMQ command) instead of shared memory, since shared
    memory can't cross machines; everything else is unchanged.

    Examples:
        >>> controller = FrankaRemoteController()  # spawns + owns a local server
        >>> controller.start()
        >>> controller.switch("impedance")
        >>> controller.set_freq(50)
        >>> for i in range(100):
        ...     state = controller.state
        ...     controller.set("q_desired", target)
        >>> controller.stop()

        >>> # Attach to a server already running on another machine, e.g. a
        >>> # NUC wired directly to the robot while this process reads a
        >>> # USB teleop device locally:
        >>> controller = FrankaRemoteController(host="192.168.2.200")
        >>> controller.start()
        >>> state = controller.state  # network round-trip, not shared memory
    """

    def __init__(self, robot_ip: str = None, *, home: bool = True,
                 host: str = None, port: int = None):
        if robot_ip is None:
            robot_ip = _load_last_ip()
        self.robot_ip = robot_ip
        self._home = home
        self._host = host
        self._port = port if port is not None else DEFAULT_TCP_PORT

        self._shm = None
        self._zmq_ctx = None
        self._zmq_sock = None
        self._connected = False
        self._server_proc = None

        # Rate limiting (mirrors FrankaController)
        self._update_freq = 50.0
        self._last_update_time = {}

        # Compatibility: no-op lock for code that uses `with controller.state_lock:`
        self.state_lock = threading.Lock()

    def start(self):
        """Start the server subprocess and connect, or attach to a remote one.

        With ``host=None`` (default): spawns a 1kHz control loop in a local
        child process, then connects to it via shared memory and ZMQ. The
        subprocess terminates automatically when this script exits.

        With ``host`` set: attaches to a server already running on that
        machine (e.g. started there with ``aiofranka start-server``). No
        subprocess is spawned, and ``stop()`` will not shut the remote
        server down -- only this client's own connection.

        The robot must already be unlocked with FCI active (via
        ``aiofranka.unlock()`` or the Franka Desk web GUI). If not, the
        server subprocess will fail to start and a helpful error message
        is printed.

        Raises:
            RuntimeError: If the server fails to start, or a remote server
                isn't reachable/running.
        """
        _GREEN = "\033[32m"

        if self._host is not None:
            print(f"\n  {_BOLD}aiofranka{_RST} {_DIM}|{_RST} "
                  f"connecting to remote server {_DIM}({self._host}:{self._port}){_RST}\n")

            self._zmq_ctx = zmq.Context()
            self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
            self._zmq_sock.setsockopt(zmq.RCVTIMEO, 2000)
            self._zmq_sock.setsockopt(zmq.SNDTIMEO, 2000)
            self._zmq_sock.connect(zmq_tcp_connect_endpoint(self._host, self._port))

            resp = self._send({"cmd": "status"})
            if not resp.get("running"):
                raise RuntimeError(
                    f"Remote server at {self._host}:{self._port} is not running. "
                    f"Start it there first with: aiofranka start-server --ip {self.robot_ip}"
                )

            self._connected = True
            print(f"  {_GREEN}Ready{_RST} {_DIM}(remote, {self._host}:{self._port}){_RST}\n")
            return

        from aiofranka.server import start_subprocess

        print(f"\n  {_BOLD}aiofranka{_RST} {_DIM}|{_RST} "
              f"starting server {_DIM}({self.robot_ip}){_RST}\n")

        # Spawn server subprocess (no homing — user script controls movement)
        try:
            self._server_proc = start_subprocess(self.robot_ip)
        except RuntimeError as e:
            err = str(e).lower()
            if "unlock" in err or "fci" in err or "joint" in err or "not ready" in err:
                print(f"  {_RED}Robot is not ready.{_RST} Unlock first:\n")
                print(f"  Add to your script before {_BOLD}ctrl.start(){_RST}:")
                print(f"    {_BOLD}aiofranka.unlock(){_RST}\n")
                print(f"  Or run from the terminal right now:")
                print(f"    {_BOLD}$ aiofranka unlock{_RST}\n")
            raise

        # Register cleanup so the subprocess dies when the script exits.
        # Use stop() instead of _terminate_server() to send a graceful ZMQ
        # stop command before SIGTERM, giving the control loop time to exit
        # cleanly (avoids communication_constraints_violation reflex).
        atexit.register(self.stop)

        # Connect to the now-running server
        self._shm = StateBlock(self.robot_ip, create=False, track=False)

        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_sock.setsockopt(zmq.RCVTIMEO, 2000)
        self._zmq_sock.setsockopt(zmq.SNDTIMEO, 2000)
        self._zmq_sock.connect(zmq_endpoint_for_ip(self.robot_ip))

        resp = self._send({"cmd": "status"})
        if not resp.get("running"):
            raise RuntimeError("Server started but control loop is not running")

        self._connected = True
        print(f"  {_GREEN}Ready{_RST} {_DIM}(PID {self._server_proc.pid}){_RST}\n")

    def stop(self):
        """Stop the server subprocess and disconnect (local mode), or just
        disconnect without touching the remote control loop (``host`` mode)
        -- a network client doesn't own the remote server's lifecycle."""
        if self._connected and self._host is None:
            try:
                self._send({"cmd": "stop"})
            except Exception:
                pass
        self._disconnect()
        self._terminate_server()

    def _disconnect(self):
        if self._zmq_sock:
            self._zmq_sock.close()
            self._zmq_sock = None
        if self._zmq_ctx:
            self._zmq_ctx.term()
            self._zmq_ctx = None
        if self._shm:
            self._shm.close()
            self._shm = None
        self._connected = False

    def _terminate_server(self):
        """Terminate the server subprocess if it's still alive."""
        proc = self._server_proc
        if proc is None or not proc.is_alive():
            self._server_proc = None
            return
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        self._server_proc = None

    def _check_server_alive(self):
        """Raise if the server has died or errored."""
        if self._host is not None:
            if self._zmq_sock is None:
                raise RuntimeError("Not connected. Call start() first.")
            return  # checked per-call via get_state's own ok/error field
        if self._shm is None:
            raise RuntimeError("Not connected. Call start() first.")
        status = self._shm.read_status()
        if status == STATUS_ERROR:
            error_msg = self._shm.read_error()
            self._connected = False
            raise ServerDiedError(error_msg or "Unknown server error")

    @property
    def state(self) -> dict:
        """Read current robot state.

        Local mode: shared memory (zero-copy, no IPC overhead).
        Remote (``host``) mode: one ZMQ round-trip (``get_state`` command) --
        shared memory can't cross machines, so this is the network fallback.
        """
        self._check_server_alive()
        if self._host is not None:
            resp = self._send({"cmd": "get_state"})
            if not resp.get("ok"):
                self._connected = False
                raise ServerDiedError(resp.get("error") or "Unknown server error")
            return _unpack_state(resp["state"])
        return self._shm.read_state()

    @property
    def q_desired(self) -> np.ndarray:
        """Current desired joint positions (7,) from shared memory."""
        return self.state["q_desired"]

    @property
    def ee_desired(self) -> np.ndarray:
        """Current desired end-effector pose (4,4) from shared memory."""
        return self.state["ee_desired"]

    @property
    def torque(self) -> np.ndarray:
        """Current commanded torque (7,) from shared memory."""
        return self.state["torque"]

    @property
    def initial_qpos(self) -> np.ndarray:
        """Initial joint positions (7,) set at last switch() from shared memory."""
        return self.state["initial_qpos"]

    @property
    def initial_ee(self) -> np.ndarray:
        """Initial end-effector pose (4,4) set at last switch() from shared memory."""
        return self.state["initial_ee"]

    @property
    def running(self) -> bool:
        if not self._connected:
            return False
        try:
            resp = self._send({"cmd": "status"})
            return resp.get("running", False)
        except Exception:
            return False

    def switch(self, controller_type: str):
        """Switch control mode on the server ("impedance", "pid", "osc", "torque")."""
        self._send({"cmd": "switch", "type": controller_type})
        self._last_update_time.clear()

    def set_freq(self, freq: float):
        """Set update frequency for rate-limited set() calls (Hz)."""
        self._update_freq = freq
        self._send({"cmd": "set_freq", "freq": freq})

    def set(self, attr: str, value):
        """
        Rate-limited setter for controller attributes.

        Sends value to server and sleeps to maintain frequency from set_freq().

        Args:
            attr: "q_desired", "ee_desired", or "torque"
            value: numpy array
        """
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq
        value = np.asarray(value, dtype=np.float64)

        if attr not in self._last_update_time:
            self._last_update_time[attr] = current_time
            self._send_set(attr, value)
            time.sleep(dt)
            self._last_update_time[attr] = current_time + dt
            return

        target_time = self._last_update_time[attr] + dt
        self._send_set(attr, value)

        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        self._last_update_time[attr] = target_time

    def step(self, attr: str, value) -> dict:
        """
        Combined set() + state-read, for tight loops that need both every
        tick (e.g. GELLO teleop). In host (``host=``) mode this folds what
        would otherwise be two separate blocking ZMQ round-trips -- set()'s
        own round-trip, then a separate ``state`` read -- into one, which
        roughly halves per-tick network latency without changing what's
        set or what's returned. In local mode it's exactly ``set()``
        followed by reading ``state`` (shared memory either way, no round
        trip to save) -- composed here purely so callers have one API
        regardless of mode; local-mode behavior is unchanged.

        Args:
            attr: "q_desired", "ee_desired", or "torque" (same as set()).
            value: numpy array.

        Returns:
            Freshly-read state dict (same keys/shapes as the ``state`` property).
        """
        if self._host is None:
            self.set(attr, value)
            return self.state

        self._check_server_alive()
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq
        value = np.asarray(value, dtype=np.float64)

        if attr not in self._last_update_time:
            self._last_update_time[attr] = current_time
            resp = self._send_step(attr, value)
            time.sleep(dt)
            self._last_update_time[attr] = current_time + dt
        else:
            target_time = self._last_update_time[attr] + dt
            resp = self._send_step(attr, value)
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_update_time[attr] = target_time

        if not resp.get("ok"):
            self._connected = False
            raise ServerDiedError(resp.get("error") or "Unknown server error")
        return _unpack_state(resp["state"])

    def _send_step(self, attr: str, value: np.ndarray) -> dict:
        return self._send({
            "cmd": "step",
            "attr": attr,
            "value": value.astype(np.float64).tobytes(),
            "shape": list(value.shape),
        })

    def move(self, qpos=None, max_velocity=None, max_acceleration=None, max_jerk=None):
        """
        Move to target joint position. Blocks until complete.

        Args:
            qpos: Target joint positions (7,). Default: home position.
            max_velocity: Optional per-joint velocity limit [rad/s].
                          Default: aiofranka's built-in default (10 rad/s).
            max_acceleration: Optional per-joint acceleration limit [rad/s²].
                          Default: aiofranka's built-in default (5 rad/s²).
            max_jerk: Optional per-joint jerk limit [rad/s³].
                          Default: aiofranka's built-in default (1 rad/s³).
        """
        if qpos is None:
            qpos = [0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853]

        qpos = np.asarray(qpos, dtype=np.float64)
        msg = {"cmd": "move", "qpos": qpos.tobytes()}
        if max_velocity is not None:
            msg["max_velocity"] = max_velocity
        if max_acceleration is not None:
            msg["max_acceleration"] = max_acceleration
        if max_jerk is not None:
            msg["max_jerk"] = max_jerk
        resp = self._send(msg)
        if not resp.get("ok"):
            raise RuntimeError(f"Move failed: {resp.get('error')}")

        move_id = resp["move_id"]
        while True:
            resp = self._send({"cmd": "move_status", "move_id": move_id})
            if resp.get("done"):
                if resp.get("error"):
                    raise RuntimeError(f"Move error: {resp['error']}")
                return
            time.sleep(0.05)

    def initialize(self):
        """Re-initialize controller state to current robot position."""
        self._send({"cmd": "initialize"})

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('robot_ip', 'state_lock'):
            super().__setattr__(name, value)
            return

        if name in _REMOTE_ATTRS and self.__dict__.get('_connected', False):
            self._send_set(name, value if not isinstance(value, np.ndarray) else value)
        else:
            super().__setattr__(name, value)

    # --- Internal ---

    def _send_set(self, attr: str, value):
        if isinstance(value, np.ndarray):
            msg = {
                "cmd": "set",
                "attr": attr,
                "value": value.astype(np.float64).tobytes(),
                "shape": list(value.shape),
            }
        else:
            msg = {"cmd": "set", "attr": attr, "value": value}
        self._send(msg)

    def _send(self, msg: dict) -> dict:
        if self._zmq_sock is None:
            raise RuntimeError("Not connected. Call start() first.")
        try:
            self._zmq_sock.send(msgpack.packb(msg, use_bin_type=True))
            raw = self._zmq_sock.recv()
            return msgpack.unpackb(raw, raw=False)
        except zmq.error.Again:
            # ZMQ timed out — check if the server died with an error
            server_error = None
            if self._shm is not None:
                try:
                    status = self._shm.read_status()
                    if status == STATUS_ERROR:
                        server_error = self._shm.read_error()
                except Exception:
                    pass

            # Fallback: if shm is gone but server process is dead, we know it crashed
            if not server_error and self._server_proc is not None:
                if not self._server_proc.is_alive():
                    server_error = "Server process died unexpectedly"

            if server_error:
                self._connected = False
                raise ServerDiedError(server_error) from None

            raise ConnectionError(
                "Server not responding (timeout). Check if server is running."
            ) from None

    def __del__(self):
        self._disconnect()
        self._terminate_server()


def _unpack_state(packed: dict) -> dict:
    """Inverse of the server's get_state packing (see server.py CommandHandler)."""
    state = {}
    for k, v in packed.items():
        if isinstance(v, dict) and v.get("__nd__"):
            state[k] = np.frombuffer(v["data"], dtype=np.float64).reshape(v["shape"]).copy()
        else:
            state[k] = v
    return state


def _load_last_ip() -> str:
    config_path = os.path.expanduser("~/.aiofranka/config.json")
    try:
        with open(config_path) as f:
            return json.load(f).get("last_ip", DEFAULT_IP)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_IP