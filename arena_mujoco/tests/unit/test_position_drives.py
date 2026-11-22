import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import pytest
import sensor_msgs.msg

from arena_mujoco import context, control
from arena_mujoco.drives import joint_drives
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.SpawnUrdf import RobotEntry, _attach_robot, _initial_positions, _mimics, _mujoco_compat

_URDF = """<robot name="arm">
  <link name="base"><inertial><mass value="5"/><inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="upper"><inertial><origin xyz="0.2 0 0"/><mass value="2"/><inertia ixx="0.01" iyy="0.03" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.05 0.05"/></geometry></collision></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/><axis xyz="0 0 1"/><limit lower="-2.5" upper="2.5" effort="26" velocity="1.95"/><dynamics damping="40" friction="0"/></joint>
  <ros2_control name="arm_system" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="shoulder"><command_interface name="position"/><state_interface name="position"/><param name="kp">2600.0</param><param name="kd">130.0</param><param name="effort">26.0</param></joint>
  </ros2_control>
</robot>
"""

P_PARAMS = '<param name="kp">2600.0</param><param name="kd">130.0</param><param name="effort">26.0</param>'


@pytest.fixture
def urdf(tmp_path: Path) -> Path:
    path = tmp_path / 'arm.urdf'
    path.write_text(_URDF)
    return path


def _rig(urdf: Path) -> tuple[SceneStore, control._Robot]:
    compat, command_interfaces, _ = _mujoco_compat(str(urdf))
    robot_spec = mujoco.MjSpec.from_file(compat)
    store = SceneStore()
    store.ensure_env(0)
    context.set_scene_store(store)
    entry = RobotEntry(
        robot_model='arm',
        tf_prefix='env_0/arm/',
        base_frame='base',
        prim_name='arm_base',
        localization=False,
        joint_names=('shoulder',),
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
        root = _attach_robot(spec, store.asset_cache(0), robot_spec, 'arm_', 'base', (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), False)
        control._pre_compile(0, spec, root, entry, ['shoulder'])
        return root

    store.add_body(0, build)
    store.recompile(0)
    robot = control._Robot(0, ('shoulder',), control._pending.pop('arm_base'), 'arm_base')
    robot.mjcf_joints = control._pending_mjcf.pop('arm_base')
    robot.resolve(store.get_model(0))
    return store, robot


def test_a_drive_reads_its_gains_and_effort_from_the_joint_params(urdf: Path):
    drive = joint_drives(str(urdf))['shoulder']
    assert (drive.kp, drive.kd, drive.effort, drive.kind) == (2600.0, 130.0, 26.0, 'position')


def test_an_effort_commanded_joint_reads_as_an_effort_drive(tmp_path: Path):
    path = tmp_path / 'effort.urdf'
    path.write_text(_URDF.replace('<command_interface name="position"/>', '<command_interface name="effort"/>'))
    assert joint_drives(str(path))['shoulder'].kind == 'effort'


def test_a_joint_without_params_has_no_drive(tmp_path: Path):
    path = tmp_path / 'bare.urdf'
    path.write_text(_URDF.replace(P_PARAMS, ''))
    assert joint_drives(str(path)) == {}


def _effort_rig(tmp_path: Path) -> tuple[object, control._Robot]:
    path = tmp_path / 'effort.urdf'
    path.write_text(_URDF.replace('<command_interface name="position"/>', '<command_interface name="effort"/>'))
    return _rig(path)


def _effort_command(*values: float) -> sensor_msgs.msg.JointState:
    command = sensor_msgs.msg.JointState()
    command.name = ['shoulder']
    command.effort = list(values)
    return command


def test_an_effort_joint_holds_its_spawn_pose_until_a_finite_effort_arrives(tmp_path: Path):
    store, robot = _effort_rig(tmp_path)
    model = store.get_model(0)
    hold = model.actuator(robot.actuators['shoulder'][0]).id
    robot.stage_effort(_effort_command(float('nan')))
    robot.apply_and_publish(0.0)
    assert float(model.actuator_gainprm[hold, 0]) == pytest.approx(2600.0)
    robot.stage_effort(_effort_command(0.0))
    robot.apply_and_publish(0.0)
    assert float(model.actuator_gainprm[hold, 0]) == 0.0
    assert list(model.actuator_biasprm[hold, :3]) == [0.0, 0.0, 0.0]


def test_an_effort_joint_is_driven_by_the_commanded_effort_clamped_to_its_limit(tmp_path: Path):
    store, robot = _effort_rig(tmp_path)
    model = store.get_model(0)
    motor = model.actuator(robot.actuators['shoulder'][2]).id
    assert list(model.actuator_forcerange[motor]) == [-26.0, 26.0]
    robot.stage_effort(_effort_command(1000.0))
    robot.apply_and_publish(0.0)
    store.step(200)
    assert float(store.get_data(0).actuator_force[motor]) == pytest.approx(26.0)


def test_a_position_driven_joint_swings_to_its_target_near_its_speed_limit(urdf: Path):
    store, robot = _rig(urdf)
    command = sensor_msgs.msg.JointState()
    command.name = ['shoulder']
    command.position = [1.5]
    robot.stage_position(command)
    robot.apply_and_publish(0.0)
    model = store.get_model(0)
    store.step(int(1.5 / model.opt.timestep))
    qpos = float(store.get_data(0).qpos[model.joint(robot.mjcf_joints['shoulder']).qposadr[0]])
    assert qpos == pytest.approx(1.5, abs=0.05)


def test_a_position_driven_joint_loses_its_urdf_damping(urdf: Path):
    compat, _, _ = _mujoco_compat(str(urdf))
    dynamics = ET.parse(compat).getroot().find("joint[@name='shoulder']/dynamics")
    assert dynamics is None or 'damping' not in dynamics.attrib
