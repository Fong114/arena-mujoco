import pytest

from arena_mujoco.scene import DEFAULT_CAPTURE_SIZE, SceneStore, parse_capture_size


@pytest.mark.parametrize(
    ('value', 'size'),
    [
        ('1280x720', (1280, 720)),
        (' 640X480 ', (640, 480)),
        ('', DEFAULT_CAPTURE_SIZE),
        ('  ', DEFAULT_CAPTURE_SIZE),
    ],
)
def test_capture_size_parses_wxh_and_empty_means_default(value: str, size: tuple[int, int]):
    assert parse_capture_size(value) == size


@pytest.mark.parametrize('value', ['1920', '1920x', '0x1080', '1920x0', 'axb', '-1x5', '1x2x3'])
def test_capture_size_rejects_anything_but_positive_wxh(value: str):
    with pytest.raises(ValueError, match=r"viewport\.resolution: expected WxH, got "):
        parse_capture_size(value)


def test_env_model_carries_the_store_capture_size():
    store = SceneStore((800, 600))
    store.ensure_env(0)
    model = store.get_model(0)
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (800, 600)


def test_default_store_keeps_the_default_capture_size(store: SceneStore):
    model = store.get_model(0)
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == DEFAULT_CAPTURE_SIZE
