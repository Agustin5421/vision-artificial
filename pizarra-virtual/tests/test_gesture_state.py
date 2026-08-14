from src.gesture_state import GestureStabilizer, mode_for_count


def test_stable_count_starts_at_zero():
    stabilizer = GestureStabilizer(required_frames=3)
    assert stabilizer.stable_count == 0


def test_stabilizes_after_required_consecutive_frames():
    stabilizer = GestureStabilizer(required_frames=3)
    stabilizer.update(1)
    stabilizer.update(1)
    assert stabilizer.stable_count == 0  # todavía no llegó a 3 frames seguidos
    stabilizer.update(1)
    assert stabilizer.stable_count == 1


def test_mode_for_count_maps_known_counts():
    assert mode_for_count(0, previous_mode="DRAW") == "PAUSE"
    assert mode_for_count(1, previous_mode="PAUSE") == "DRAW"
    assert mode_for_count(2, previous_mode="DRAW") == "MOVE"
    assert mode_for_count(5, previous_mode="DRAW") == "CLEAR"


def test_mode_for_count_keeps_previous_mode_for_undefined_counts():
    assert mode_for_count(3, previous_mode="DRAW") == "DRAW"
    assert mode_for_count(4, previous_mode="MOVE") == "MOVE"


def test_single_frame_noise_does_not_change_stable_count():
    stabilizer = GestureStabilizer(required_frames=3)
    for _ in range(3):
        stabilizer.update(1)
    assert stabilizer.stable_count == 1

    stabilizer.update(2)  # ruido de un solo frame
    assert stabilizer.stable_count == 1

    stabilizer.update(1)
    assert stabilizer.stable_count == 1
