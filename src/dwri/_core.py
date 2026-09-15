from collections.abc import Callable
from typing import Literal

import polars as pl

ScoreStrategy = Literal[
    "linear",
    "sqrt",
    "log1p_abs",
]
NormalisationStrategy = Literal[
    "zscore",
    "quantile",
]


def _calc_weighted_reception_importance(
    df: pl.DataFrame,
    col_tokens: str,
    col_weight: str,
    score_fn: Callable[[pl.Expr], pl.Expr],
) -> pl.DataFrame:
    df_score = (
        df.explode(col_tokens, empty_as_null=False)
        .group_by(col_tokens)
        .agg(
            score=score_fn(pl.col(col_weight)).sum(),
        )
    )
    df_nt = (
        df.select(
            pl.col(col_tokens).list.unique(),
        )
        .explode(col_tokens, empty_as_null=False)
        .group_by(col_tokens)
        .agg(
            nt=pl.len(),
        )
    )
    N = df.height

    df_wri = (
        df_score.join(
            df_nt,
            on=col_tokens,
            how="inner",
        )
        .with_columns(
            penalty=(pl.lit(N) / (pl.col("nt") + 1)).log() + 1,
        )
        .with_columns(
            wri=pl.col("score") / pl.col("score").sum() * pl.col("penalty"),
        )
        .select([col_tokens, "nt", "score", "penalty", "wri"])
    )
    return df_wri


def compute_dwri(
    df_signal: pl.DataFrame,
    df_noise: pl.DataFrame,
    col_tokens: str = "tokens",
    col_weight: str = "weight",
    score_strategy: ScoreStrategy = "log1p_abs",
    score_fn: Callable[[pl.Expr], pl.Expr] | None = None,
    normalisation_strategy: NormalisationStrategy = "zscore",
    normalisation_fn: Callable[[pl.Expr], pl.Expr] | None = None,
) -> tuple[dict[str, float], pl.DataFrame]:
    """
    Compute Differentially Weighted Reception Importance (DWRI) for each token.

    For each corpus, a Weighted Reception Importance (WRI) score is computed
    per token. The signal and noise WRI scores are then subtracted to produce
    a raw differential, which is normalised to yield the final DWRI.

    Parameters
    ----------
    df_signal : pl.DataFrame
        Signal corpus — documents representing the target phenomenon.
        Must contain ``col_tokens`` (list of str) and ``col_weight`` (numeric).
    df_noise : pl.DataFrame
        Noise corpus — background or baseline documents used to cancel out
        tokens that are merely common rather than distinctive.
        Must contain the same ``col_tokens`` and ``col_weight`` columns.
    col_tokens : str, optional
        Name of the column containing per-document token lists.
        Defaults to ``"tokens"``.
    col_weight : str, optional
        Name of the column containing per-document reception weights
        (e.g. upvotes, citations). Defaults to ``"weight"``.
    score_strategy : {"log1p_abs", "linear", "sqrt"}, optional
        Built-in function used to transform weights before aggregation.
        ``"log1p_abs"`` applies ``log(1 + |w - 1|)``, compressing large
        weights while preserving direction. Ignored when ``score_fn`` is
        provided. ``"log1p_abs"`` is recommended for social media data,
        e.g., Reddit. Defaults to ``"log1p_abs"``.
    score_fn : Callable[[pl.Expr], pl.Expr] or None, optional
        Custom weight transformation as a Polars expression function.
        Overrides ``score_strategy`` when provided. Defaults to ``None``.
    normalisation_strategy : {"zscore", "quantile"}, optional
        Built-in normalisation applied to the raw DWRI scores.
        ``"zscore"`` centres and scales by standard deviation;
        ``"quantile"`` maps ranks to ``[0, 1]``. Ignored when
        ``normalisation_fn`` is provided. Defaults to ``"zscore"``.
    normalisation_fn : Callable[[pl.Expr], pl.Expr] or None, optional
        Custom normalisation as a Polars expression function.
        Overrides ``normalisation_strategy`` when provided. Defaults to ``None``.

    Returns
    -------
    dict_dwri : dict[str, float]
        Mapping of keyword → normalised DWRI score, sorted descending.
    df_dwri : pl.DataFrame
        Full per-token DataFrame with intermediate columns from both corpora
        (``nt``, ``score``, ``penalty``, ``wri`` for signal and noise),
        plus ``dwri_raw`` and ``dwri``, sorted by ``dwri`` descending.
    """
    if score_fn is None:
        strategies = {
            "linear": lambda w: w,
            "sqrt": lambda w: w.sqrt(),
            "log1p_abs": lambda w: (pl.lit(1) + (w - 1).abs()).log(),
        }
        if score_strategy not in strategies:
            raise ValueError(f"Unknown score strategy: {score_strategy!r}. Expected one of: {', '.join(strategies)}.")
        score_fn = strategies[score_strategy]

    if normalisation_fn is None:
        strategies = {
            "zscore": lambda wri: (wri - wri.mean()) / wri.std(),
            "quantile": lambda wri: (wri.rank(method="average") - 1) / (wri.count() - 1),
        }
        if normalisation_strategy not in strategies:
            raise ValueError(f"Unknown normalisation strategy: {normalisation_strategy!r}. Expected one of: {', '.join(strategies)}.")
        normalisation_fn = strategies[normalisation_strategy]

    df_signal = _calc_weighted_reception_importance(
        df_signal,
        col_tokens,
        col_weight,
        score_fn,
    ).rename(lambda col: f"{col}_signal" if col != col_tokens else col)

    df_noise = _calc_weighted_reception_importance(
        df_noise,
        col_tokens,
        col_weight,
        score_fn,
    ).rename(lambda col: f"{col}_noise" if col != col_tokens else col)

    if df_signal.height < 2 or df_noise.height < 2:
        raise ValueError(f"`compute_dwri()` requires at least 2 unique tokens in each corpus after preprocessing; got {df_signal.height} signal and {df_noise.height} noise tokens.")

    df_dwri = (
        df_signal.join(
            df_noise,
            on=col_tokens,
            how="left",
        )
        .fill_null(0)
        .with_columns(
            dwri_raw=pl.col("wri_signal") - pl.col("wri_noise"),
        )
        .with_columns(
            dwri=normalisation_fn(pl.col("dwri_raw")),
        )
        .sort("dwri", descending=True)
        .rename({col_tokens: "keyword"})
    )

    if not df_dwri.select(pl.col("dwri").is_finite().fill_null(False).all()).item():
        raise ValueError("DWRI normalisation produced non-finite values; check the input corpora, weights, and normalisation function.")

    dict_dwri = {row["keyword"]: row["dwri"] for row in df_dwri.select(["keyword", "dwri"]).to_dicts()}

    return dict_dwri, df_dwri
