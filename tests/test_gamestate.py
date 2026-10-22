import numpy as np
import pandas as pd

from pitchstate.gamestate import Variant, assemble, ball_positions, to_predictions

# Pixels -> metres: x_m = x / 10 - 50, y_m = y / 10 - 30
H = np.array([[0.1, 0, -50], [0, 0.1, -30], [0, 0, 1]], dtype=float)
FRAMES = 12


def _dets():
    rows = []
    for f in range(1, FRAMES + 1):
        rows.append({"frame": f, "x1": 100 + 5 * f, "y1": 100, "x2": 140 + 5 * f, "y2": 200, "score": 0.9,
                     "class_name": "player"})  # fmt: skip
        rows.append({"frame": f, "x1": 600, "y1": 300, "x2": 640, "y2": 400, "score": 0.9, "class_name": "referee"})
        if f not in (5, 6):  # the ball goes missing for two frames
            rows.append({"frame": f, "x1": 300 + f, "y1": 395, "x2": 310 + f, "y2": 405, "score": 0.8,
                         "class_name": "ball"})  # fmt: skip
    return pd.DataFrame(rows)


NO_SAM = pd.DataFrame(columns=["frame", "obj_id", "prob", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "area"])


def test_assemble_maps_people_to_metres_with_roles():
    Hs = {f: H for f in range(1, FRAMES + 1)}
    state = assemble(_dets(), NO_SAM, Hs, None, {}, FRAMES, v=Variant(smooth=1))
    assert set(state["role"]) == {"player", "referee"}
    referee = state[state["role"] == "referee"]
    assert np.allclose(referee[["x_m", "y_m"]].to_numpy(), [12.0, 10.0])  # (620, 400) px, box bottom-centre
    assert referee["team"].isna().all() and referee["jersey"].isna().all()
    player = state[state["role"] == "player"].sort_values("frame")
    assert np.allclose(np.diff(player["x_m"]), 0.5)  # 5 px per frame -> 0.5 m per frame


def test_assemble_without_homographies_is_empty():
    assert assemble(_dets(), NO_SAM, {}, None, None, FRAMES).empty


def test_ball_positions_fill_short_gaps():
    Hs = {f: H for f in range(1, FRAMES + 1)}
    ball = ball_positions(_dets(), Hs).set_index("frame")
    assert list(ball.index) == list(range(1, FRAMES + 1))  # frames 5-6 interpolated
    assert np.isclose(ball.loc[5, "x_m"], (305 + 5) / 10 - 50)
    assert np.isclose(ball.loc[1, "y_m"], 405 / 10 - 30)


def test_to_predictions_uses_gsr_image_ids():
    state = pd.DataFrame([{"frame": 3, "track_id": 1, "role": "player", "team": "left", "jersey": "9",
                           "x_m": 1.0, "y_m": 2.0, "score": 0.8}])  # fmt: skip
    (pred,) = to_predictions(state, {3: "2021000003"})
    assert pred["image_id"] == "2021000003"
    assert pred["attributes"] == {"role": "player", "team": "left", "jersey": "9"}
    assert pred["bbox_pitch"]["x_bottom_middle"] == 1.0
