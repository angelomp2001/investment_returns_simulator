'''
8/17/2025 chatgpt
'''

from __future__ import annotations

import time
from pathlib import Path
from datetime import date
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)



# ----------------------------
# Input normalization & cache
# ----------------------------

def _standardize_inputs(
    symbols: Iterable[str] | str,
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
) -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
    """Ensure symbols is a list and dates are Timestamps."""
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = list(symbols)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return symbols, start_ts, end_ts


def _find_cached_combined(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    combined_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Look for any 'symbols_data_*.csv' in combined_dir containing all requested symbols.
    If found and not empty over [start:end], return the slice; else None.
    """
    paths = sorted(combined_dir.glob("symbols_data_*.csv"), reverse=True)
    for p in paths:
        try:
            df = pd.read_csv(p, index_col="Date", parse_dates=True)
        except Exception:
            continue
        if set(symbols).issubset(df.columns):
            if df.index.min() <= start and df.index.max() >= end:            
                subset = df.loc[start:end, symbols]
                if not subset.empty:
                    return subset
    return None


# ----------------------------
# Symbol storage helpers
# ----------------------------

def _symbol_file_path(symbol: str, symbol_dir: Path) -> Path:
    return symbol_dir / f"{symbol}.csv"


def _ensure_dirs(symbol_dir: Path, combined_dir: Path) -> None:
    symbol_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)


def _symbol_data_existing_dates(symbol: str, symbol_dir: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return (min_date, max_date) for a symbol CSV, or (None, None) if no file."""
    fp = _symbol_file_path(symbol, symbol_dir)
    if not fp.exists():
        return None, None
    try:
        sdf = pd.read_csv(fp, index_col="Date", parse_dates=True)
        if sdf.empty:
            return None, None
        return sdf.index.min(), sdf.index.max()
    except Exception:
        return None, None


# ----------------------------
# Missing-range logic
# ----------------------------

def _compute_missing_ranges(
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    existing_start: Optional[pd.Timestamp],
    existing_end: Optional[pd.Timestamp],
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], str]:
    """
    Reproduce your 0–6 cases and return a list of missing (start, end) ranges plus a case label.
    Returned ranges are inclusive endpoints aligned to how yfinance is called (end + 1 day).
    """
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    if existing_start is None or existing_end is None:
        # Case 0: no existing data
        ranges.append((requested_start, requested_end))
        return ranges, "case_0_no_existing"

    s, e = requested_start, requested_end
    es, ee = existing_start, existing_end

    if s < es and e < es:
        # Case 1: s---e___es===ee
        ranges.append((s, es))
        return ranges, "case_1_before_existing"

    if s < es and e <= ee:
        # Case 2: s---es===e===ee
        ranges.append((s, es))
        return ranges, "case_2_overlap_left"

    if s >= es and e <= ee:
        # Case 3: es===s===e===ee (fully covered)
        return ranges, "case_3_fully_covered"

    if s >= es and e > ee:
        # Case 4: es===s===ee---e
        ranges.append((ee + pd.Timedelta(days=1), e))
        return ranges, "case_4_extend_right"

    if s > ee and e > ee:
        # Case 5: es===ee___s---e
        ranges.append((ee + pd.Timedelta(days=1), e))
        return ranges, "case_5_after_existing"

    if s < es and e > ee:
        # Case 6: s---es===ee---e (two gaps)
        ranges.append((s, es))
        ranges.append((ee + pd.Timedelta(days=1), e))
        return ranges, "case_6_brackets_existing"

    # Fallback (shouldn't happen)
    return ranges, "case_unknown"


# ----------------------------
# File masking & updates
# ----------------------------

def _write_mask_preserving_existing(
    symbol: str,
    mask_start: pd.Timestamp,
    mask_end: pd.Timestamp,
    existing_start: Optional[pd.Timestamp],
    existing_end: Optional[pd.Timestamp],
    symbol_dir: Path,
) -> None:
    """
    Create a mask CSV spanning min(start, existing_start) .. max(end, existing_end),
    keeping any existing 'Close' values.
    """
    symbol_fp = _symbol_file_path(symbol, symbol_dir)

    # Compute inclusive range spanning incoming+existing
    lo = min(filter(pd.notnull, [mask_start, existing_start]))
    hi = max(filter(pd.notnull, [mask_end, existing_end]))
    date_index = pd.date_range(lo, hi, freq="D")

    mask = pd.DataFrame(index=date_index)
    mask["Close"] = np.nan

    if symbol_fp.exists():
        try:
            prev = pd.read_csv(symbol_fp, index_col="Date", parse_dates=True)
            mask.loc[prev.index, "Close"] = prev["Close"]
        except Exception:
            pass

    mask.to_csv(symbol_fp, index_label="Date")


def _download_yfinance_close(
    symbol: str,
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    """
    Download close prices for [start, end_inclusive] (yfinance end is exclusive).
    Returns a DataFrame indexed by Date with a 'Close' column.
    """
    params = {
        "start": start,
        "end": end_inclusive + pd.Timedelta(days=1),  # yfinance end is exclusive
        "auto_adjust": True,
        "rounding": True,
        "group_by": "Symbol",
    }
    df = yf.download(symbol, **params)

    if df.empty:
        return pd.DataFrame(columns=["Close"])

    # Flatten possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Expected form: ('SYMBOL', 'Open'/'Close'/...)
        # We only want the second level (price field)
        df.columns = [c[1] for c in df.columns]

    # Keep only Close if present
    if "Close" not in df.columns:
        return pd.DataFrame(columns=["Close"])

    out = df[["Close"]].copy()
    out.index.name = "Date"
    return out


def _merge_download_into_symbol_file(
    symbol: str,
    downloaded: pd.DataFrame,
    symbol_dir: Path,
) -> None:
    """
    Merge downloaded 'Close' into the symbol CSV; assumes the CSV exists (mask created earlier).
    """
    fp = _symbol_file_path(symbol, symbol_dir)
    if not fp.exists():
        # Create empty shell if mask wasn't written for some reason
        idx = downloaded.index if not downloaded.empty else pd.DatetimeIndex([], name="Date")
        pd.DataFrame(index=idx, columns=["Close"]).to_csv(fp, index_label="Date")

    current = pd.read_csv(fp, index_col="Date", parse_dates=True)
    if not downloaded.empty:
        current.loc[downloaded.index, "Close"] = downloaded["Close"].values
    current.sort_index(inplace=True)
    current.to_csv(fp, index_label="Date")


def _update_all_symbols_metadata(
    symbol: str,
    new_data_start: Optional[pd.Timestamp],
    new_data_end: Optional[pd.Timestamp],
    existing_start: Optional[pd.Timestamp],
    existing_end: Optional[pd.Timestamp],
    meta_path: Path,
) -> None:
    """
    Maintain a CSV (index='symbol') with columns ['start_date','end_date'] reflecting total coverage.
    """
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["start_date", "end_date"]

    if meta_path.exists():
        meta = pd.read_csv(meta_path, index_col="symbol", parse_dates=cols)
    else:
        meta = pd.DataFrame(columns=cols)
        meta.index.name = "symbol"

    # Compute min/max while tolerating None
    start_candidates = [d for d in [new_data_start, existing_start] if pd.notnull(d)]
    end_candidates = [d for d in [new_data_end, existing_end] if pd.notnull(d)]

    start_val = min(start_candidates) if start_candidates else pd.NaT
    end_val = max(end_candidates) if end_candidates else pd.NaT

    meta.loc[symbol, "start_date"] = start_val
    meta.loc[symbol, "end_date"] = end_val
    meta.to_csv(meta_path, index_label="symbol")


def _delete_symbol_from_metadata_and_file(symbol: str, meta_path: Path, symbol_dir: Path) -> None:
    """Handle the 'empty download' case: remove symbol from metadata and delete its CSV."""
    # Update metadata
    if meta_path.exists():
        try:
            meta = pd.read_csv(meta_path, index_col="symbol", parse_dates=["start_date", "end_date"])
            if symbol in meta.index:
                meta = meta.drop(symbol, axis=0)
                # meta.to_csv(meta_path, index_label="symbol")
        except Exception:
            pass
    # Delete symbol file
    fp = _symbol_file_path(symbol, symbol_dir)
    if fp.exists():
        try:
            pass #fp.unlink()
        except Exception:
            pass


# ----------------------------
# Assembly helpers
# ----------------------------

def _read_symbol_series(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol_dir: Path,
) -> pd.Series:
    """Read symbol CSV and return Close series for [start:end]."""
    fp = _symbol_file_path(symbol, symbol_dir)
    sdf = pd.read_csv(fp, index_col="Date", parse_dates=True)
    sdf.index = pd.to_datetime(sdf.index)
    sdf.sort_index(inplace=True)
    return sdf.loc[start:end, "Close"].copy()


def _assemble_dataframe(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Combine per-symbol Close series into a single DataFrame."""
    df = pd.DataFrame(series_dict)
    df.index.name = "Date"
    return df


def _save_combined_snapshot(df: pd.DataFrame, combined_dir: Path) -> Path:
    """Save combined dataframe to symbols_data/symbols_data_<timestamp>.csv and return the path."""
    ts = f"{date.today().strftime('%Y-%m-%d')}"
    out_path = combined_dir / f"symbols_data_{ts}.csv"
    df.to_csv(out_path, index_label="Date")
    return out_path


# ----------------------------
# Orchestrator
# ----------------------------

def get_symbol_data(
    symbols: List[str] | str,
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
    *,
    symbol_dir: Path | str = "symbols",
    combined_dir: Path | str = "symbols_data",
    metadata_path: Path | str = "Equity Universe - Symbols.csv",
) -> pd.DataFrame:
    """
    High-level pipeline:
      1) Try to reuse a cached combined file.
      2) For each symbol, compute missing ranges vs. local CSV.
      3) For each missing range, write a mask, download from yfinance, merge, and update metadata.
      4) Read all symbols and assemble the final DataFrame; save a combined snapshot.
    """
    symbol_dir = Path(symbol_dir)
    combined_dir = Path(combined_dir)
    metadata_path = Path(metadata_path)
    _ensure_dirs(symbol_dir, combined_dir)

    symbols, start_ts, end_ts = _standardize_inputs(symbols, start_date, end_date)

    # 1) Fast path: reuse a cached combined file if it already covers the need.
    cached = _find_cached_combined(symbols, start_ts, end_ts, combined_dir)
    if cached is not None:
        return cached

    # 2) Update individual symbol files as needed.
    for symbol in symbols:
        sym_start, sym_end = _symbol_data_existing_dates(symbol, symbol_dir)
        req_ranges, case_label = _compute_missing_ranges(start_ts, end_ts, sym_start, sym_end)

        # Fetch only what’s missing
        for (gap_start, gap_end) in tqdm(
            req_ranges,
            desc=f"Getting {symbol}",
            unit="range",
            leave=False,
            colour="green",
            ascii=(" ", "█"),
            ncols=100,
        ):
            # Prepare file with mask to preserve old values
            _write_mask_preserving_existing(
                symbol=symbol,
                mask_start=gap_start,
                mask_end=gap_end,
                existing_start=sym_start,
                existing_end=sym_end,
                symbol_dir=symbol_dir,
            )

            # Download and merge
            dl = _download_yfinance_close(symbol, gap_start, gap_end)
            if dl.empty:
                _delete_symbol_from_metadata_and_file(symbol, metadata_path, symbol_dir)
                # If nothing comes back at all, break; no point trying more gaps
                break
            else:
                _merge_download_into_symbol_file(symbol, dl, symbol_dir)
                # Update metadata and refresh existing range for next gaps
                _update_all_symbols_metadata(
                    symbol=symbol,
                    new_data_start=gap_start,
                    new_data_end=gap_end,
                    existing_start=sym_start,
                    existing_end=sym_end,
                    meta_path=metadata_path,
                )
                sym_start, sym_end = _symbol_data_existing_dates(symbol, symbol_dir)

    # 3) Assemble final dataframe from the per-symbol CSVs.
    series_dict: Dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            series_dict[symbol] = _read_symbol_series(symbol, start_ts, end_ts, symbol_dir)
        except Exception as e:
            # If a symbol failed entirely, fill with NaNs for the requested range
            idx = pd.date_range(start_ts, end_ts, freq="D")
            series_dict[symbol] = pd.Series(index=idx, dtype="float64", name=symbol)

    df = _assemble_dataframe(series_dict)

    # 4) Save a combined snapshot and return.
    _save_combined_snapshot(df, combined_dir)
    return df
