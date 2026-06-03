"""
Basic usage of DWRI with example data.

Demonstrates how to use compute_dwri to extract keywords distinctively
associated with a signal corpus relative to a noise corpus, weighted by
per-document reception signals (e.g. post upvotes).
"""

import polars as pl

from dwri import compute_dwri

# ---------------------------------------------------------------------------
# 1. Prepare data
#
# Each row is one document. "tokens" is a list of pre-tokenised strings;
# "weight" is a per-document reception signal (e.g. upvotes, citation count).
# ---------------------------------------------------------------------------

df_signal = pl.DataFrame({
    "tokens": [
        ["climate", "crisis", "flood", "warming"],
        ["flood", "disaster", "climate"],
        ["warming", "sea", "level", "rise"],
        ["renewable", "energy", "climate", "policy"],
    ],
    "weight": [350.0, 210.0, 95.0, 420.0],
})

df_noise = pl.DataFrame({
    "tokens": [
        ["weather", "rain", "flood", "storm"],
        ["news", "report", "today", "update"],
        ["energy", "price", "oil", "market"],
        ["policy", "government", "budget"],
    ],
    "weight": [80.0, 40.0, 110.0, 60.0],
})

# ---------------------------------------------------------------------------
# 2. Default settings (log1p_abs score, zscore normalisation)
# ---------------------------------------------------------------------------

dict_dwri, df_dwri = compute_dwri(df_signal, df_noise)

print("Top keywords (log1p_abs + zscore):")
for keyword, score in list(dict_dwri.items())[:5]:
    print(f"  {keyword:<20} {score:+.4f}")

print()
print("Full result table (top 5 rows):")
print(df_dwri.head(5))
print()

# ---------------------------------------------------------------------------
# 3. Custom score and normalisation functions
# ---------------------------------------------------------------------------

dict_custom, _ = compute_dwri(
    df_signal,
    df_noise,
    score_fn=lambda w: w,                                            # linear
    normalisation_fn=lambda x: (x - x.min()) / (x.max() - x.min()),  # min-max
)

print("Top keywords (linear score + min-max normalisation):")
for keyword, score in list(dict_custom.items())[:5]:
    print(f"  {keyword:<20} {score:.4f}")
