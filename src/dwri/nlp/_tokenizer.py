import argparse
import csv
import gc
from collections import Counter
from pathlib import Path

import polars as pl
import spacy
from localspelling.spelling_converter import get_dictionary
from spacy.language import Language
from tqdm import tqdm

from dwri.nlp import _utils


def _extract_lemmas(
    content: pl.Series,
    nlp: Language,
    dict_us2gb: dict[str, str],
    n_process: int = -1,
) -> tuple[list[list[str]], list[list[str]], Counter, Counter, Counter]:
    count = 0
    propns = Counter()
    alnums = Counter()
    nonalnums = Counter()
    lemmas_all_list = []
    lemmas_list = []
    for document in tqdm(nlp.pipe(content.to_list(), n_process=n_process), total=content.len()):
        lemmas_all = []
        lemmas_nonstop = []
        for token in document:
            # skip non-meaningful tokens
            if token.is_punct or token.is_space or token.like_url or token.like_num or token.like_email:
                continue
            # token should contain at least one alphabetic character to be meaningful
            if not any(char.isalpha() for char in token.text):
                continue

            # lower case and strip punctuations
            lemma = _utils.strip_punctuation(token.lemma_.lower())
            # skip 1-letter lemmas
            if len(lemma) <= 1:
                continue
            # apply US to GB spelling conversion, if applicable
            lemma = dict_us2gb.get(lemma, lemma)

            # collect proper nouns separately
            if token.pos_ == "PROPN":
                propns[_utils.strip_punctuation(token.text.lower())] += 1
            # collect alphanumeric (all characters are alphanumeric) and non-alphanumeric
            if lemma.isalnum():
                alnums[lemma] += 1
            else:
                nonalnums[lemma] += 1

            lemmas_all.append(lemma)
            if not token.is_stop:
                lemmas_nonstop.append(lemma)

        lemmas_all_list.append(lemmas_all)
        lemmas_list.append(lemmas_nonstop)

        count += 1
        if count % 500_000 == 0:
            print(f"{count:_}: {len(alnums):_} alnums, {len(nonalnums):_} non-alnums, {len(propns):_} propn")

    print(f"{count:_}: {len(alnums):_} alnums, {len(nonalnums):_} non-alnums, {len(propns):_} propn")
    return lemmas_list, lemmas_all_list, alnums, nonalnums, propns


def tokenize(
    df: pl.DataFrame | str | Path,
    col_text: str = "text",
    n_process: int = -1,
) -> tuple[pl.DataFrame, Counter, Counter, Counter]:
    """
    Normalise and tokenise text, returning lemmas and frequency counters.

    Loads a spaCy ``en_core_web_lg`` pipeline (NER and parser disabled) and a
    US→GB spelling dictionary, then for each document: unescapes HTML entities,
    normalises unicode (NFKC), fixes known bad spellings, and extracts
    lower-cased, punctuation-stripped lemmas. Stop words are kept in
    ``tokens_all`` and excluded from ``tokens``. Null rows in the source column
    are preserved as null in the output.

    Parameters
    ----------
    df : pl.DataFrame or str or Path
        Input data as a DataFrame or a path to a Parquet file. Only
        ``col_text`` is read; all other columns are ignored.
    col_text : str, optional
        Name of the column containing raw text. Defaults to ``"text"``.
    n_process : int, optional
        Number of worker processes passed to ``nlp.pipe``. ``-1`` uses all
        available cores. Set to ``1`` when calling from a Jupyter notebook
        or any context where ``multiprocessing`` spawn is problematic.
        Defaults to ``-1``.

    Returns
    -------
    df : pl.DataFrame
        DataFrame with columns ``tokens`` (non-stop-word lemmas) and
        ``tokens_all`` (all lemmas), each row a list of strings.
    alnums : Counter
        Frequency counter of fully alphanumeric lemmas.
    nonalnums : Counter
        Frequency counter of lemmas containing non-alphanumeric characters.
    propns : Counter
        Frequency counter of proper-noun surface forms.
    """
    nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
    dict_us2gb: dict[str, str] = get_dictionary("gb")

    if isinstance(df, (str, Path)):
        df = pl.scan_parquet(df).select([col_text]).collect()

    is_null = df[col_text].is_null()
    normalised_text = df.select(
        normalised_text=pl.col(col_text)
        # fill null with empty string to avoid pipeline crashing on nulls
        .fill_null("")
        # unescape HTML entities
        .map_elements(_utils.fully_unescape_html, return_dtype=pl.Utf8, skip_nulls=True)
        # normalise unicode to NFKC
        .str.normalize("NFKC")
        # fix some known bad spellings that cause problems for spacy lemmatization
        .str.replace_many(
            ["‘", "’", "“", "”", '"', "+", "-", "(", ")", "/", "\\"],
            ["'", "'", "'", "'", "'", " ", " ", " ", " ", " ", "\\"],
            ascii_case_insensitive=True,
        )
        # replace all -ization* with -isation*
        .str.replace_all(r"(?i)(?<=\w)ization(?=\w*)", "isation", literal=False),
    ).to_series()

    del df
    gc.collect()
    print(f"Processing {normalised_text.len():_} entries")

    lemmas_list, lemmas_all_list, alnums, nonalnums, propns = _extract_lemmas(
        normalised_text,
        nlp,
        dict_us2gb,
        n_process=n_process,
    )

    df = (
        pl.DataFrame(
            {
                "is_null": is_null,
                "tokens": pl.Series(lemmas_list),
                "tokens_all": pl.Series(lemmas_all_list),
            }
        )
        .with_columns(
            # restore null: if original text is null, set results to null as well
            tokens=pl.when(pl.col("is_null")).then(None).otherwise(pl.col("tokens")),
            tokens_all=pl.when(pl.col("is_null")).then(None).otherwise(pl.col("tokens_all")),
        )
        .select(["tokens", "tokens_all"])
    )

    return df, alnums, nonalnums, propns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract lemmas from text data")
    parser.add_argument("input_file", type=str, help="Path to input Parquet file (polars) containing text data")
    parser.add_argument("col_text", type=str, help="Name of the column containing text data")
    parser.add_argument("--save_path", type=Path, default=Path("output"), help="Directory to save output files")
    parser.add_argument("--n_process", type=int, default=-1, help="Number of processes to use for spacy parallel processing")
    args = parser.parse_args()

    args.save_path.mkdir(parents=True, exist_ok=True)

    df, alnums, nonalnums, propns = tokenize(
        args.input_file,
        args.col_text,
        n_process=args.n_process,
    )
    output_file = args.save_path / "tokens.parquet"
    df.write_parquet(output_file)
    print(f"Saved {output_file}: {output_file.stat().st_size / 1024**2:.2f} MB")

    for counter, name in zip([alnums, nonalnums, propns], ["alnums", "nonalnums", "propns"]):
        output_file = args.save_path / f"{name}.csv"
        with output_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "count"])
            for lemma, count in counter.most_common():
                writer.writerow([lemma, count])
        print(f"Saved {output_file}: {output_file.stat().st_size / 1024**2:.2f} MB")
