from chicken_tracker.neural import input_shape, prepare
from chicken_tracker.tracking import TrackConfig


def test_input_shape_matches_rect_letterbox():
    assert input_shape(1920, 1536, 1280) == (1024, 1280)   # the camera: no grey bars at all
    assert input_shape(1920, 1536, 640) == (512, 640)
    assert input_shape(1920, 1080, 1280) == (736, 1280)    # 720 padded up to a multiple of 32


def test_gpu_when_asked_or_not_on_the_mac_gpu():
    assert prepare(TrackConfig(model="x.pt", engine="gpu"), "v.mp4", 1920, 1536).engine == "gpu"
    assert prepare(TrackConfig(model="x.pt", device="cpu"), "v.mp4", 1920, 1536).engine == "gpu"
    assert prepare(TrackConfig(model="missing.pt"), "v.mp4", 1920, 1536).engine == "gpu"
