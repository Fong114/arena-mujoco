"""AttachPrim and DetachPrim service handlers: carry a world prim with a robot's hand."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from arena_mujoco_msgs.srv import AttachPrim, DetachPrim

from arena_mujoco import control, hooks
from arena_mujoco.context import get_scene_store

from .utils import Service

if TYPE_CHECKING:
    from arena_mujoco.services.SpawnUrdf import RobotEntry


def _hand_body(env_id: int, robot: str, joint: str) -> str | None:
    """Body id of the hand that carries the gripper joint of robot, None when robot or joint is unknown."""
    wiring = control.find_robot(env_id, robot)
    model = get_scene_store().get_model(env_id)
    if wiring is None or model is None or joint not in wiring.mjcf_joints:
        return None
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wiring.mjcf_joints[joint])
    if joint_id < 0:
        return None
    return model.body(int(model.body_parentid[model.jnt_bodyid[joint_id]])).name


def _attach_prim_cb(request: AttachPrim.Request, response: AttachPrim.Response) -> AttachPrim.Response:
    env_id = int(request.env_id)
    hand = _hand_body(env_id, request.robot, request.joint)
    logger = hooks.get_node().get_logger()
    if hand is None:
        wiring = control.find_robot(env_id, request.robot)
        known = sorted(wiring.mjcf_joints) if wiring is not None else None
        logger.warning(f"AttachPrim: no hand for joint {request.joint!r} of robot {request.robot!r}, its joints: {known}")
        response.attached = ''
        return response
    store = get_scene_store()
    attached = store.attach(env_id, hand, list(request.objects), float(request.reach))
    if attached is None:
        poses = {name: store.get_body_pose(env_id, name) for name in (hand, *request.objects)}
        logger.warning(f"AttachPrim: nothing within {request.reach:.2f} m of hand {hand!r}, gaps: {store.gaps(env_id, hand, list(request.objects), 10.0)}, poses: {poses}")
    response.attached = attached or ''
    return response


def _detach_prim_cb(request: DetachPrim.Request, response: DetachPrim.Response) -> DetachPrim.Response:
    env_id = int(request.env_id)
    hand = _hand_body(env_id, request.robot, request.joint)
    response.released = [] if hand is None else get_scene_store().detach(env_id, hand)
    return response


def _despawn(env_id: int, robot_entry: RobotEntry) -> None:
    get_scene_store().detach_under(env_id, robot_entry.prim_name)


hooks.register_despawn(_despawn)

attach_prim_service = Service(srv_type=AttachPrim, srv_name='mujoco/AttachPrim', callback=_attach_prim_cb)
detach_prim_service = Service(srv_type=DetachPrim, srv_name='mujoco/DetachPrim', callback=_detach_prim_cb)

__all__ = ['attach_prim_service', 'detach_prim_service']
