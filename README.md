# DWRI — Differentially Weighted Reception Importance

DWRI is a keyword extraction method that identifies terms distinctively associated with a *signal* corpus relative to a *noise* (baseline) corpus, weighted by per-document reception signals such as upvotes or citation counts. It was proposed in [citation pending — see below].

## Installation

**Core**:

Requires only [polars](https://pola.rs) for performance optimisation.

```bash
pip install dwri
```

**With tokenizer** (adds spaCy-based English lemmatisation):

```bash
pip install "dwri[nlp]"
python -m spacy download en_core_web_lg
```

## Quick Start

```python
import polars as pl
from dwri import compute_dwri

df_signal = pl.DataFrame({
    "tokens": [["climate", "crisis", "flood"], ["flood", "warming"]],
    "weight": [120.0, 45.0],  # e.g. post upvotes
})
df_noise = pl.DataFrame({
    "tokens": [["news", "flood", "rain"], ["climate", "report"]],
    "weight": [30.0, 15.0],
})

dict_dwri, df_dwri = compute_dwri(df_signal, df_noise)
print(list(dict_dwri.items())[:5])  # top 5 keywords
```

## How It Works

For each corpus, DWRI first computes a **Weighted Reception Importance (WRI)** score per token:

$$\text{WRI}(t) = \frac{\sum_{d:\, t \in D_d} f(w_d)}{\sum_{t'} \sum_{d:\, t' \in D_d} f(w_d)} \times \left(\log\frac{N}{n_t + 1} + 1\right)$$

where $f$ is a weight-transformation function (default: $f(w) = \log(1 + |w - 1|)$), $w_d$ is the reception weight of document $d$, $n_t$ is the number of documents containing token $t$, and $N$ is the total number of documents. The first factor measures how important (popularity, reception) the token is within the corpus. The second factor is a smoothed modifier that balances the token's rarity against its ubiquity.

The raw differential is then:

$$\text{DWRI}_{\text{raw}}(t) = \text{WRI}_{\text{signal}}(t) - \text{WRI}_{\text{noise}}(t)$$

and normalised (default: z-score) to produce the final DWRI score. Tokens with high DWRI scores are both reception-weighted and distinctive to the signal corpus.

## API Reference

### `compute_dwri`

```python
from dwri import compute_dwri

dict_dwri, df_dwri = compute_dwri(
    df_signal,                        # pl.DataFrame
    df_noise,                         # pl.DataFrame
    col_tokens="tokens",              # column of List[str]
    col_weight="weight",              # column of numeric
    score_strategy="log1p_abs",       # "linear" | "sqrt" | "log1p_abs"
    score_fn=None,                    # custom Callable[[pl.Expr], pl.Expr]
    normalisation_strategy="zscore",  # "zscore" | "quantile"
    normalisation_fn=None,            # custom Callable[[pl.Expr], pl.Expr]
)
```

Returns `(dict[str, float], pl.DataFrame)`. The dict maps keyword → normalised DWRI score (sorted descending). The DataFrame contains intermediate columns (`nt`, `score`, `penalty`, `wri` for both corpora) alongside `dwri_raw` and `dwri`.

### `tokenize`

Optional, requires `dwri[nlp]`.

```python
from dwri.nlp import tokenize

df_tokens, alnums, nonalnums, propns = tokenize(
    df,               # pl.DataFrame or path to Parquet file
    col_text="text",  # column of raw text
    n_process=-1,     # worker processes (-1 = all cores; use 1 in notebooks)
)
```

Returns a DataFrame with `tokens` (non-stop lemmas) and `tokens_all` (all lemmas), plus three `Counter` objects for vocabulary inspection.

Also available as a CLI (after `pip install "dwri[nlp]"`):

```bash
python -m dwri.nlp._tokenizer input.parquet "col_text" --save_path output/
```

## Citation

> [Paper under review. Citation will be added upon publication.]

## Acknowledgements

Hideaki Takeda, for his supervision and support during the development of this research. ([ORCID](https://orcid.org/0000-0002-2909-7163))

## License

Apache-2.0 © Yanming He
