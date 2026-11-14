import mujoco
import numpy as np
import pytest

from arena_mujoco.scene import SceneStore

_HAND = 'env0_hand'


def _hand(spec: mujoco.MjSpec) -> mujoco.MjsBody:
    body = spec.worldbody.add_body(name=_HAND, pos=[0.0, 0.0, 1.0])
    body.add_freejoint().name = 'hand_free'
    body.gravcomp = 1.0
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=1.0)
    finger = body.add_body(name='env0_finger', pos=[0.0, 0.0, -0.1])
    finger.gravcomp = 1.0
    finger.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.05], mass=0.1)
    return body


def _box(name: str, pos: tuple[float, float, float], free: bool = False):
    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name=name, pos=list(pos))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=1.0)
        if free:
            body.add_freejoint().name = f'{name}_free'
            body.gravcomp = 1.0
        return body

    return build


def _move_hand(store: SceneStore, pos: tuple[float, float, float]) -> None:
    model, data = store.get_model(0), store.get_data(0)
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hand_free')]
    data.qpos[adr : adr + 3] = pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


@pytest.fixture
def scene(store: SceneStore) -> SceneStore:
    store.add_body(0, _hand)
    store.add_body(0, _box('crate', (0.0, 0.0, 0.7)))
    store.add_body(0, _box('far', (5.0, 0.0, 0.7)))
    store.recompile(0)
    return store


def test_the_nearest_prim_within_reach_is_attached(scene: SceneStore):
    assert scene.attach(0, _HAND, ['far', 'crate'], 0.2) == 'crate'


def test_nothing_is_attached_beyond_reach(scene: SceneStore):
    assert scene.attach(0, _HAND, ['far'], 0.2) is None


def test_an_unknown_hand_or_prim_attaches_nothing(scene: SceneStore):
    assert scene.attach(0, 'ghost', ['crate'], 1.0) is None
    assert scene.attach(0, _HAND, ['ghost'], 1.0) is None
    assert scene.attach(7, _HAND, ['crate'], 1.0) is None


def test_a_prim_is_attached_under_its_wire_name(scene: SceneStore):
    scene.add_body(0, _box('env_0/Obstacles/box-1', (0.0, 0.0, 0.8)))
    scene.recompile(0)
    assert scene.attach(0, _HAND, ['env_0/Obstacles/box-1'], 0.2) == 'env_0/Obstacles/box-1'


def test_a_static_prim_follows_the_hand_through_steps(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    _move_hand(scene, (2.0, 1.0, 1.5))
    scene.step(5)
    pos, _ = scene.get_body_pose(0, 'crate')
    assert pos == pytest.approx((2.0, 1.0, 0.7 + 0.5), abs=1e-3)


def test_a_prim_turns_with_the_hand(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    model, data = scene.get_model(0), scene.get_data(0)
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hand_free')]
    data.qpos[adr + 3 : adr + 7] = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    mujoco.mj_forward(model, data)
    scene.step(2)
    pos, quat = scene.get_body_pose(0, 'crate')
    assert quat == pytest.approx((np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)), abs=1e-3)
    assert pos == pytest.approx((0.0, 0.0, 0.7), abs=1e-3)


def test_a_free_body_follows_the_hand_without_falling(scene: SceneStore):
    scene.add_body(0, _box('ball', (0.0, 0.0, 0.8), free=True))
    scene.recompile(0)
    assert scene.attach(0, _HAND, ['ball'], 0.2) == 'ball'
    _move_hand(scene, (1.0, 0.0, 2.0))
    scene.step(50)
    pos, _ = scene.get_body_pose(0, 'ball')
    assert pos == pytest.approx((1.0, 0.0, 1.8), abs=1e-3)


def test_a_carried_prim_stops_colliding_until_detached(scene: SceneStore):
    model = scene.get_model(0)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate')
    geoms = np.flatnonzero(model.geom_bodyid == body)
    before = (model.geom_contype[geoms].copy(), model.geom_conaffinity[geoms].copy())
    scene.attach(0, _HAND, ['crate'], 0.2)
    assert not model.geom_contype[geoms].any()
    assert not model.geom_conaffinity[geoms].any()
    assert not model.body_contype[body]
    scene.detach(0, _HAND)
    assert (model.geom_contype[geoms] == before[0]).all()
    assert (model.geom_conaffinity[geoms] == before[1]).all()
    assert model.body_contype[body] == before[0].max()


def test_a_detached_prim_stays_where_the_hand_left_it(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    _move_hand(scene, (2.0, 0.0, 1.0))
    scene.step(2)
    assert scene.detach(0, _HAND) == ['crate']
    _move_hand(scene, (-3.0, 0.0, 1.0))
    scene.step(5)
    pos, _ = scene.get_body_pose(0, 'crate')
    assert pos == pytest.approx((2.0, 0.0, 0.7), abs=1e-3)


def test_detaching_an_empty_hand_releases_nothing(scene: SceneStore):
    assert scene.detach(0, _HAND) == []
    assert scene.detach(9, _HAND) == []


def test_a_carried_static_prim_keeps_its_pose_across_a_recompile(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    _move_hand(scene, (2.0, 0.0, 1.0))
    scene.step(2)
    scene.add_body(0, _box('late', (9.0, 9.0, 0.1)))
    scene.recompile(0)
    mujoco.mj_forward(scene.get_model(0), scene.get_data(0))
    assert scene.get_body_pose(0, 'crate')[0] == pytest.approx((2.0, 0.0, 0.7), abs=1e-3)
    scene.detach(0, _HAND)
    scene.recompile(0)
    mujoco.mj_forward(scene.get_model(0), scene.get_data(0))
    assert scene.get_body_pose(0, 'crate')[0] == pytest.approx((2.0, 0.0, 0.7), abs=1e-3)


def test_a_carried_prim_is_still_carried_after_a_recompile(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    scene.add_body(0, _box('late', (9.0, 9.0, 0.1)))
    scene.recompile(0)
    _move_hand(scene, (0.0, 3.0, 1.0))
    scene.step(2)
    assert scene.get_body_pose(0, 'crate')[0] == pytest.approx((0.0, 3.0, 0.7), abs=1e-3)
    model = scene.get_model(0)
    assert not model.geom_contype[model.geom_bodyid == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate')].any()


def test_removing_a_carried_prim_forgets_it(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    assert scene.remove(0, 'crate')
    scene.recompile(0)
    assert scene.detach(0, _HAND) == []


def test_removing_the_hand_releases_the_prim_with_its_collision(scene: SceneStore):
    scene.attach(0, _HAND, ['crate'], 0.2)
    assert scene.remove(0, _HAND)
    scene.recompile(0)
    model = scene.get_model(0)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'crate')
    assert model.geom_contype[model.geom_bodyid == body].all()
    assert scene.detach(0, _HAND) == []


def test_detaching_under_a_prefix_releases_every_hand_below_it(scene: SceneStore):
    scene.add_body(0, _box('stool', (0.0, 0.0, 0.85)))
    scene.recompile(0)
    scene.attach(0, _HAND, ['crate'], 0.2)
    scene.attach(0, 'env0_finger', ['stool'], 0.2)
    assert sorted(scene.detach_under(0, 'env0_')) == ['crate', 'stool']
    assert scene.detach_under(0, 'env0_') == []


def test_a_prim_already_carried_is_not_attached_twice(scene: SceneStore):
    assert scene.attach(0, _HAND, ['crate'], 0.2) == 'crate'
    assert scene.attach(0, _HAND, ['crate'], 0.2) is None
