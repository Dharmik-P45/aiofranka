"""
aiofranka: Asyncio-based Franka Robot Control

A high-level Python library for controlling Franka Emika robots using asyncio.
Combines pylibfranka for real-time control with MuJoCo for kinematics/dynamics.

Main Components:
    RobotInterface: Low-level robot interface (real or simulation)
    FrankaController: High-level asyncio controller with multiple modes
    FrankaLockUnlock: Client for robot authentication and brake control

Quick Example:
    >>> import asyncio
    >>> from aiofranka import RobotInterface, FrankaController
    >>> 
    >>> async def main():
    ...     robot = RobotInterface("172.16.0.2")
    ...     controller = FrankaController(robot)
    ...     await controller.start()
    ...     await controller.move()  # Move to home
    ...     await controller.stop()
    >>> 
    >>> asyncio.run(main())

For detailed documentation, see README.md and USAGE_GUIDE.md
"""

from aiofranka.async_utils import asyncify, async_input, CudaInferenceThread, mpify
from aiofranka.remote import FrankaRemoteController, ServerDiedError
from aiofranka.remote_v2 import FrankaRemoteControllerV2

__version__ = "0.1.0"
__all__ = ["FrankaRemoteController", "FrankaRemoteControllerV2", "ServerDiedError", "asyncify", "async_input", "CudaInferenceThread", "mpify"]

# Server/sim-only imports (need mujoco + pylibfranka) - optional, so a
# client-only environment (e.g. running a trained policy, which only needs
# FrankaRemoteController's shared-memory + ZMQ client) doesn't need mujoco
# installed at all. Same pattern as the Robotiq guard below.
try:
    from aiofranka.controller import FrankaController
    from aiofranka.robot import RobotInterface
    from aiofranka.server import start, stop, lock, unlock, set_configuration
    _HAS_SERVER = True
except ImportError:
    _HAS_SERVER = False

if _HAS_SERVER:
    __all__.extend(["RobotInterface", "FrankaController", "start", "stop", "lock", "unlock", "set_configuration"])

# Optional gripper support - only import if dependencies are available
try:
    from aiofranka.gripper import GripperController, RobotiqGripperInterface, create_gripper
    from aiofranka.gripper_remote import GripperRemoteController
    _HAS_ROBOTIQ = True
except ImportError:
    _HAS_ROBOTIQ = False

if _HAS_ROBOTIQ:
    __all__.extend(["GripperController", "GripperRemoteController", "RobotiqGripperInterface", "create_gripper"])