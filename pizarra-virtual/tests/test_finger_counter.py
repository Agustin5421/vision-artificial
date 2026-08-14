from src.finger_counter import count_extended_fingers


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_hand(overrides=None):
    """21 landmarks for a closed fist (all tips below/near their PIP/base), then override."""
    points = {
        0: (0.5, 0.9),   # wrist
        1: (0.45, 0.85), 2: (0.42, 0.8), 3: (0.4, 0.75), 4: (0.4, 0.72),   # thumb
        5: (0.47, 0.7), 6: (0.47, 0.65), 7: (0.47, 0.68), 8: (0.47, 0.7),  # index
        9: (0.5, 0.7), 10: (0.5, 0.65), 11: (0.5, 0.68), 12: (0.5, 0.7),   # middle
        13: (0.53, 0.7), 14: (0.53, 0.65), 15: (0.53, 0.68), 16: (0.53, 0.7),  # ring
        17: (0.56, 0.72), 18: (0.56, 0.68), 19: (0.56, 0.7), 20: (0.56, 0.72),  # pinky
    }
    points.update(overrides or {})
    return [Landmark(*points[i]) for i in range(21)]


def test_closed_fist_counts_zero():
    hand = make_hand()
    assert count_extended_fingers(hand) == 0


def test_open_hand_counts_five():
    hand = make_hand({
        4: (0.30, 0.72),   # thumb tip far from palm ref (17)
        8: (0.47, 0.3),    # index tip above pip
        12: (0.5, 0.3),    # middle tip above pip
        16: (0.53, 0.3),   # ring tip above pip
        20: (0.56, 0.3),   # pinky tip above pip
    })
    assert count_extended_fingers(hand) == 5


def test_only_index_extended_counts_one():
    hand = make_hand({8: (0.47, 0.3)})
    assert count_extended_fingers(hand) == 1


def test_index_and_middle_extended_counts_two():
    hand = make_hand({8: (0.47, 0.3), 12: (0.5, 0.3)})
    assert count_extended_fingers(hand) == 2
