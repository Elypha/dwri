import polars as pl
from dwri import compute_dwri
from polars.testing import assert_frame_equal


def test_compute_dwri_golden():
    signal = pl.DataFrame({
        "tokens": [
            ["alpha", "alpha", "shared"],
            ["beta", "shared"],
            ["alpha", "gamma", "shared"],
            ["gamma", "signal_only"],
            [],
        ],
        "weight": [11.0, -9.0, 1.0, 4.0, 100.0],
    })
    noise = pl.DataFrame({
        "tokens": [
            ["alpha", "shared", "noise_only"],
            ["alpha", "alpha", "beta"],
            ["beta", "shared", "shared"],
            ["gamma", "noise_only"],
            [],
        ],
        "weight": [6.0, 2.0, -4.0, 1.0, 50.0],
    })
    expected = pl.DataFrame({
        "keyword": ["signal_only", "gamma", "alpha", "beta", "shared"],
        "nt_signal": [1, 2, 2, 1, 3],
        "score_signal": [1.3862943611198906, 1.3862943611198906, 4.795790545596741, 2.3978952727983707, 4.795790545596741],
        "penalty_signal": [1.916290731874155, 1.5108256237659907, 1.5108256237659907, 1.916290731874155, 1.2231435513142097],
        "wri_signal": [0.1799574124856805, 0.14188049101718814, 0.4908258566926457, 0.31127518138055876, 0.39736583228937167],
        "nt_noise": [0, 1, 2, 2, 2],
        "score_noise": [0.0, 0.0, 3.1780538303479453, 2.4849066497880004, 5.375278407684165],
        "penalty_noise": [0.0, 1.916290731874155, 1.5108256237659907, 1.5108256237659907, 1.5108256237659907],
        "wri_noise": [0.0, 0.0, 0.3742389536596965, 0.2926158316383423, 0.632978128850964],
        "dwri_raw": [0.1799574124856805, 0.14188049101718814, 0.11658690303294922, 0.018659349742216447, -0.23561229656159238],
        "dwri": [0.8101682437453499, 0.5827760780528934, 0.4317249024837748, -0.1530901919342261, -1.6715790323477921],
    })

    dict_dwri, df_dwri = compute_dwri(signal, noise)

    assert_frame_equal(df_dwri, expected.select("keyword", "dwri", "dwri_raw", pl.exclude("keyword", "dwri", "dwri_raw")), check_dtypes=False, rel_tol=1e-12, abs_tol=1e-12)
    assert list(dict_dwri) == expected["keyword"].to_list()
    assert_frame_equal(pl.DataFrame({"dwri": list(dict_dwri.values())}), expected.select("dwri"), rel_tol=1e-12, abs_tol=1e-12)
