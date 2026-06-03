import math

import polars as pl

from dwri import compute_dwri


def test_basic_returns_correct_types_and_structure():
    df_signal = pl.DataFrame({
        "tokens": [["apple", "banana", "apple"], ["banana", "cherry"]],
        "weight": [10.0, 5.0],
    })
    df_noise = pl.DataFrame({
        "tokens": [["apple", "date"], ["elderberry"]],
        "weight": [3.0, 2.0],
    })
    dict_dwri, df_dwri = compute_dwri(df_signal, df_noise)

    assert isinstance(dict_dwri, dict)
    assert isinstance(df_dwri, pl.DataFrame)
    expected_cols = {
        "keyword", "nt_signal", "score_signal", "penalty_signal", "wri_signal",
        "nt_noise", "score_noise", "penalty_noise", "wri_noise",
        "dwri_raw", "dwri",
    }
    assert expected_cols.issubset(set(df_dwri.columns))
    scores = df_dwri["dwri"].to_list()
    assert scores == sorted(scores, reverse=True)
    assert set(dict_dwri.keys()) == set(df_dwri["keyword"].to_list())


def test_token_only_in_signal_gets_finite_score():
    df_signal = pl.DataFrame({
        "tokens": [["exclusive", "shared"], ["exclusive"]],
        "weight": [8.0, 4.0],
    })
    df_noise = pl.DataFrame({
        "tokens": [["shared", "other"], ["other"]],
        "weight": [3.0, 2.0],
    })
    dict_dwri, _ = compute_dwri(df_signal, df_noise)

    assert "exclusive" in dict_dwri
    score = dict_dwri["exclusive"]
    assert not math.isnan(score)
    assert not math.isinf(score)
    assert dict_dwri["exclusive"] > dict_dwri["shared"]


def test_custom_score_fn_and_normalisation_fn():
    df_signal = pl.DataFrame({
        "tokens": [["alpha", "beta"], ["alpha", "gamma"]],
        "weight": [4.0, 2.0],
    })
    df_noise = pl.DataFrame({
        "tokens": [["beta", "delta"], ["delta"]],
        "weight": [1.0, 1.0],
    })

    custom_score = lambda w: w * 2
    custom_norm = lambda x: (x - x.min()) / (x.max() - x.min())

    dict_dwri, df_dwri = compute_dwri(
        df_signal,
        df_noise,
        score_fn=custom_score,
        normalisation_fn=custom_norm,
    )

    assert isinstance(dict_dwri, dict)
    scores = df_dwri["dwri"].to_list()
    assert min(scores) >= -1e-9
    assert max(scores) <= 1.0 + 1e-9
