from pathlib import Path

import mujoco
import pytest

import numpy as np

from arena_mujoco import context, control
from arena_mujoco.drives import joint_drives
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.SpawnUrdf import RobotEntry, _attach_robot, _initial_positions, _mimics, _mujoco_compat

_INERTIAL = '<inertial><mass value="{m}"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>'
_BOX = '<collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>'
_URDF = f"""<robot name="r">
  <link name="base">{_INERTIAL.format(m=5)}</link>
  <link name="payload">{_INERTIAL.format(m=17)}{_BOX}</link>
  <link name="right_finger">{_INERTIAL.format(m=0.1)}</link>
  <link name="left_finger">{_INERTIAL.format(m=0.1)}</link>
  <joint name="lift" type="prismatic"><parent link="base"/><child link="payload"/><axis xyz="0 0 1"/><limit lower="0" upper="0.5" effort="2000" velocity="1"/></joint>
  <joint name="right_finger_joint" type="prismatic"><parent link="base"/><child link="right_finger"/><origin xyz="0.3 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="0.04" effort="20" velocity="1"/></joint>
  <joint name="left_finger_joint" type="prismatic"><parent link="base"/><child link="left_finger"/><origin xyz="-0.3 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="0.04" effort="20" velocity="1"/><mimic joint="right_finger_joint" multiplier="-1" offset="0.01"/></joint>
  <ros2_control name="r_system" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="lift"><command_interface name="position"/><state_interface name="position"/><param name="kp">83385.0</param><param name="kd">2381.2</param><param name="effort">2000.0</param></joint>
    <joint name="right_finger_joint"><command_interface name="position"/><state_interface name="position"/><param name="kp">490.5</param><param name="kd">14.0</param><param name="effort">20.0</param></joint>
    <joint name="left_finger_joint"><command_interface name="position"/><state_interface name="position"/><param name="kp">490.5</param><param name="kd">14.0</param><param name="effort">20.0</param></joint>
  </ros2_control>
</robot>
"""
_JOINTS = ('lift', 'right_finger_joint', 'left_finger_joint')


@pytest.fixture
def rig(tmp_path: Path):
    urdf = tmp_path / 'r.urdf'
    urdf.write_text(_URDF)
    compat, command_interfaces, _ = _mujoco_compat(str(urdf))
    robot_spec = mujoco.MjSpec.from_file(compat)
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    entry = RobotEntry(
        robot_model='r',
        tf_prefix='env_0/r/',
        base_frame='base',
        prim_name='r_base',
        localization=False,
        joint_names=_JOINTS,
        joint_states_topic='',
        cmd_vel_topic='',
        odom_topic='',
        command_interfaces=command_interfaces,
        urdf_sensors={},
        sensors=(),
        initial_positions=_initial_positions(str(urdf)),
        mimics=_mimics(str(urdf)),
        drives=joint_drives(str(urdf)),
    )

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        root = _attach_robot(spec, store.asset_cache(0), robot_spec, 'r_', 'base', (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), False)
        control._pre_compile(0, spec, root, entry, list(_JOINTS))
        return root

    store.add_body(0, build)
    store.recompile(0)
    robot = control._Robot(0, _JOINTS, control._pending.pop('r_base'), 'r_base')
    robot.mjcf_joints = control._pending_mjcf.pop('r_base')
    robot.resolve(store.get_model(0))
    return store, robot


def _qpos(store: SceneStore, robot: control._Robot, joint: str) -> float:
    model = store.get_model(0)
    return float(store.get_data(0).qpos[model.joint(robot.mjcf_joints[joint]).qposadr[0]])


def test_a_heavy_position_driven_prismatic_joint_settles_on_its_target_under_gravity(rig):
    store, robot = rig
    robot.stage_position(_command({'lift': 0.3}))
    robot.apply_and_publish(0.0)
    store.step(2000)
    assert _qpos(store, robot, 'lift') == pytest.approx(0.3, abs=0.005)


def test_a_mimic_joint_follows_its_leader_and_appears_in_the_joint_state(rig):
    store, robot = rig
    robot.stage_position(_command({'right_finger_joint': 0.006}))
    robot.apply_and_publish(0.0)
    store.step(2000)
    leader = _qpos(store, robot, 'right_finger_joint')
    follower = _qpos(store, robot, 'left_finger_joint')
    assert leader == pytest.approx(0.006, abs=0.001)
    assert follower == pytest.approx(0.01 - leader, abs=0.001)
    msg = robot._joint_state_msg(store.get_model(0), store.get_data(0), 1.0, ('left_finger_joint',))
    assert list(msg.name) == ['left_finger_joint']
    assert msg.position[0] == pytest.approx(follower)
    assert len(msg.velocity) == len(msg.name)


def test_a_mimic_joint_gets_no_actuator(rig):
    _, robot = rig
    assert 'left_finger_joint' not in robot.actuators


_LEGGED = f"""<robot name="q">
  <link name="body">{_INERTIAL.format(m=8)}<collision><geometry><box size="0.4 0.2 0.1"/></geometry></collision></link>
  <link name="leg">{_INERTIAL.format(m=4)}<collision><origin xyz="0 0 -0.15"/><geometry><box size="0.3 0.3 0.1"/></geometry></collision></link>
  <joint name="knee" type="revolute"><parent link="body"/><child link="leg"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="200" velocity="10"/></joint>
  <ros2_control name="q_system" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="knee"><command_interface name="position"/><state_interface name="position"/><param name="kp">20000.0</param><param name="kd">1000.0</param><param name="effort">200.0</param></joint>
  </ros2_control>
</robot>
"""


def test_a_floating_base_rests_its_whole_weight_on_its_contacts(tmp_path: Path):
    urdf = tmp_path / 'q.urdf'
    urdf.write_text(_LEGGED)
    compat, command_interfaces, _ = _mujoco_compat(str(urdf))
    robot_spec = mujoco.MjSpec.from_file(compat)
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    entry = RobotEntry(
        robot_model='q',
        tf_prefix='env_0/q/',
        base_frame='body',
        prim_name='q_body',
        localization=True,
        joint_names=('knee',),
        joint_states_topic='',
        cmd_vel_topic='',
        odom_topic='',
        command_interfaces=command_interfaces,
        urdf_sensors={},
        sensors=(),
        drives=joint_drives(str(urdf)),
    )

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.1])
        root = _attach_robot(spec, store.asset_cache(0), robot_spec, 'q_', 'body', (0.0, 0.0, 0.25), (1.0, 0.0, 0.0, 0.0), True)
        control._pre_compile(0, spec, root, entry, ['knee'])
        return root

    store.add_body(0, build)
    store.recompile(0)
    control._pending.pop('q_body')
    control._pending_mjcf.pop('q_body')
    model, data = store.get_model(0), store.get_data(0)
    store.step(1000)
    force = np.zeros(6)
    support = 0.0
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        support += float((data.contact[i].frame.reshape(3, 3).T @ force[:3])[2])
    assert support == pytest.approx(12.0 * 9.81, rel=0.02)


def _command(positions: dict[str, float]):
    import sensor_msgs.msg

    msg = sensor_msgs.msg.JointState()
    msg.name = list(positions)
    msg.position = list(positions.values())
    return msg
