from mvfer.analytics import multiscale_entropy, session_features, shannon_entropy


def test_constant_sequence_has_no_complexity():
    assert shannon_entropy([0, 0, 0]) == 0.0
    assert multiscale_entropy([0, 0, 0])["entropy_s1"] == 0.0


def test_features_are_descriptive_and_bounded():
    features = session_features([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6])
    assert features["prop_neutral"] == 0.5
    assert features["prop_happiness"] == 0.5
    assert features["transition_rate"] == 1.0
    assert abs(features["mean_confidence"] - 0.75) < 1e-9
    assert 0.0 <= features["entropy_s1"] <= 1.0
