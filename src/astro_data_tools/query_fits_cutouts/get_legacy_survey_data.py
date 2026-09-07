#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Survey Batch Cutout Downloader
Date: May 2026
Author: Geferson Lucatelli

Downloads FITS cutouts and PSF models from the DESI Legacy Imaging Surveys
(https://www.legacysurvey.org/viewer).

Pixel scale: 0.262 arcsec/pixel (native for DR9+)

Install:
    pip install requests astropy numpy pandas joblib tqdm

Usage:
    python get_data_legacy_survey_2026.py <catalog.csv> <output_dir> [options]
    python get_data_legacy_survey_2026.py --source-name <NAME> <output_dir> [options]

Input catalog must contain RA and DEC columns (decimal degrees).
Optional columns: ID/#ID (source name), source_size (per-source angular size in arcsec),
                  D25 (arcmin diameter), R50/R90/Rad (radius columns in arcsec).

Examples:
    # DR9, g-band, 640x640 pixel cutouts
    python get_data_legacy_survey_2026.py sources.csv ./ls_out/ --bands g --size 640

    # All grz bands with PSF models, invvar, and model images
    python get_data_legacy_survey_2026.py sources.csv ./ls_out/ \\
        --bands grz --psf --invvar --model --dr 10

    # Per-source size from catalog column, factor 3
    python get_data_legacy_survey_2026.py sources.csv ./ls_out/ --bands r \\
        --use-source-size --source-size-factor 3.0

    # Auto-size from D25/R50/R90 catalog columns with extra 1.5x multiplier
    python get_data_legacy_survey_2026.py sources.csv ./ls_out/ --bands g \\
        --auto-size --size-factor 1.5

    # Retry mode: skip not-in-survey sources, only re-attempt server errors
    python get_data_legacy_survey_2026.py sources.csv ./ls_out/ --bands g --retry

    # Single source by name (resolved via SIMBAD/Sesame/NED), all bands,
    # one FITS file per band: Mrk331_g.fits, Mrk331_r.fits, Mrk331_z.fits
    python get_data_legacy_survey_2026.py --source-name Mrk331 ./output/Mrk331/ \\
        --bands ALL --psf

    # Single source at an explicit position (skips name resolution)
    python get_data_legacy_survey_2026.py --source-name Mrk331 \\
        --ra 23.5058 --dec 20.5862 ./output/Mrk331/ --bands ALL --psf
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LS_PIXSCALE_NATIVE = 0.262  # arcsec/pixel, native for DR9+
LS_BASE_URL = "https://www.legacysurvey.org/viewer"

# Bands available per data release, used to expand --bands ALL
LS_ALL_BANDS = {"10": "griz"}
LS_ALL_BANDS_DEFAULT = "grz"


def expand_bands(bands: str, data_release: str, split: bool) -> List[str]:
    """
    Turn the --bands string into the list of requests to issue.

    Each returned entry produces one FITS file. 'ALL' expands to every band of
    the data release, always one file per band. Otherwise the string is split
    into individual bands when *split* is True (single-source mode), or kept as
    a single multi-band request when False (catalog mode, one multi-plane FITS
    per source, as before).
    """
    bands = bands.strip()
    if bands.upper() == "ALL":
        return list(LS_ALL_BANDS.get(str(data_release), LS_ALL_BANDS_DEFAULT))
    if split:
        return list(bands)
    return [bands]

# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class LegacySurveyDownloader:
    """Downloads cutouts, PSF models and optional products from the Legacy Survey viewer."""

    def __init__(
        self,
        data_release: str = "10",
        pixscale: float = LS_PIXSCALE_NATIVE,
        timeout: int = 30,
        max_retries: int = 1,
    ):
        self.data_release = data_release
        self.pixscale = pixscale
        self.timeout = timeout
        self.max_retries = max_retries
        self.layer = f"ls-dr{data_release}"

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def classify_error(self, error_msg: str, status_code: Optional[int] = None) -> str:
        """Return 'not_in_survey' or 'server_error'."""
        msg = error_msg.lower()
        not_in_survey = ['no overlap', 'not in survey', 'outside coverage',
                         'no coverage', 'empty fits', 'no data']
        if any(s in msg for s in not_in_survey):
            return 'not_in_survey'
        if status_code == 404:
            return 'not_in_survey'
        if status_code in [429, 500, 502, 503]:
            return 'server_error'
        server_errs = ['timeout', 'connection', 'too many', 'rate limit',
                       'server error', 'unavailable', 'overloaded']
        if any(s in msg for s in server_errs):
            return 'server_error'
        return 'server_error'

    # ------------------------------------------------------------------
    # Core HTTP helper
    # ------------------------------------------------------------------

    def _get(self, url: str, output_path: Path) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        GET url → output_path with retries.
        Returns (ok, error_msg, http_status_code).
        Does NOT verify FITS validity; callers do that.
        """
        last_error: Optional[str] = None
        status_code: Optional[int] = None

        for _ in range(self.max_retries):
            try:
                resp = requests.get(url, timeout=self.timeout)
                status_code = resp.status_code
                resp.raise_for_status()
                output_path.write_bytes(resp.content)
                return True, None, None
            except requests.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                time.sleep(1)
            except requests.exceptions.HTTPError as exc:
                last_error = f"HTTP {exc.response.status_code}: {exc}"
                status_code = exc.response.status_code
                time.sleep(1)
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1)

        if output_path.exists():
            output_path.unlink()
        return False, last_error or "Unknown error", status_code

    # ------------------------------------------------------------------
    # Individual product downloads
    # ------------------------------------------------------------------

    def download_cutout(
        self,
        source_id: str, ra: float, dec: float, size_pixels: int,
        bands: str, output_dir: Path,
        layer_suffix: str = "",
        filename_suffix: str = "",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Download a science cutout (image, model, resid, or invvar)."""
        suffix = f"_{bands}{filename_suffix}.fits"
        output_path = output_dir / f"{source_id}{suffix}"
        if output_path.exists():
            # Earlier versions saved the whole '&invvar' response, whose first
            # HDU is the science image, so an existing invvar file may in fact
            # hold the image.  Those are re-downloaded instead of trusted.
            if filename_suffix == "_invvar" and not self._is_invvar_file(output_path):
                output_path.unlink()
            else:
                return True, None, None

        layer = self.layer + layer_suffix
        url = (
            f"{LS_BASE_URL}/cutout.fits?"
            f"ra={ra}&dec={dec}"
            f"&size={size_pixels}"
            f"&layer={layer}"
            f"&pixscale={self.pixscale}"
            f"&bands={bands}"
        )
        if filename_suffix == "_invvar":
            url += "&invvar"

        ok, err, code = self._get(url, output_path)
        if not ok:
            return False, self.classify_error(err or "Unknown error", code), err

        # Validate
        try:
            with fits.open(str(output_path)) as hdul:
                if len(hdul) == 0 or hdul[0].data is None:
                    output_path.unlink()
                    msg = "Empty FITS - outside survey coverage"
                    return False, self.classify_error(msg), msg

                if filename_suffix == "_invvar":
                    # '&invvar' does not swap the image for the weight map, it
                    # APPENDS it: HDU0 stays the science image and the
                    # inverse variance arrives as HDU1.  Saving the response
                    # untouched therefore produces a file whose first HDU is
                    # byte-identical to the plain cutout, which is what any
                    # reader picks up.  Keep only the weight plane.
                    hdu = self._select_invvar_hdu(hdul)
                    if hdu is None:
                        output_path.unlink()
                        msg = ("No INVVAR plane in the response - the layer "
                               "may not publish inverse-variance maps")
                        return False, 'no_data', msg
                    data, header = hdu.data, hdu.header.copy()
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            return False, 'server_error', str(exc)

        if filename_suffix == "_invvar":
            # Rewritten outside the context manager: the file being replaced
            # is the one that was just read.
            header["IMAGETYP"] = ("INVVAR", "Inverse variance (weight) map")
            fits.writeto(str(output_path), data, header, overwrite=True,
                         output_verify="silentfix")

        return True, None, None

    @staticmethod
    def _is_invvar_file(path: Path) -> bool:
        """True if the file on disk really holds an inverse-variance map."""
        try:
            with fits.open(str(path)) as hdul:
                for hdu in hdul:
                    if hdu.data is None:
                        continue
                    return str(hdu.header.get("IMAGETYP", "")).strip().upper() \
                        == "INVVAR"
        except Exception:      # noqa: BLE001 - unreadable means re-download
            return False
        return False

    @staticmethod
    def _select_invvar_hdu(hdul):
        """
        Pick the inverse-variance plane out of a cutout response.

        The service labels the planes with IMAGETYP ('IMAGE' / 'INVVAR'),
        which is what is trusted here; the positional fallback covers a
        response that omits the keyword.
        """
        for hdu in hdul:
            if hdu.data is None:
                continue
            if str(hdu.header.get("IMAGETYP", "")).strip().upper() == "INVVAR":
                return hdu
        data_hdus = [h for h in hdul if h.data is not None and h.data.size]
        return data_hdus[1] if len(data_hdus) > 1 else None

    def download_psf(
        self,
        source_id: str, ra: float, dec: float,
        bands: str, output_dir: Path,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Download per-band PSF model via the coadd-psf endpoint.

        NOTE: the coadd-psf endpoint always returns the PSF at the native coadd
        pixel scale (0.262 arcsec/px for DR9+), regardless of the pixscale used
        for the science cutout.  If you requested a non-native pixscale (e.g.
        0.55 arcsec/px to match S-PLUS), you must resample this PSF stamp before
        using it for convolution or model fitting.  The PSF FWHM in arcsec is
        unaffected and can be converted with: FWHM_px = FWHM_arcsec / pixscale.
        """
        output_path = output_dir / f"psf_{source_id}_{bands}.fits"
        if output_path.exists():
            return True, None, None

        url = (
            f"{LS_BASE_URL}/coadd-psf/?"
            f"ra={ra:.8f}&dec={dec:.8f}"
            f"&layer={self.layer}"
            f"&bands={bands}"
        )

        ok, err, code = self._get(url, output_path)
        if not ok:
            return False, self.classify_error(err or "Unknown error", code), err

        try:
            with fits.open(str(output_path)) as hdul:
                if len(hdul) == 0:
                    output_path.unlink()
                    msg = "Empty PSF FITS - outside survey coverage"
                    return False, self.classify_error(msg), msg
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            return False, 'server_error', str(exc)

        return True, None, None

    # ------------------------------------------------------------------
    # Per-source dispatcher
    # ------------------------------------------------------------------

    def process_source(
        self,
        source: dict,
        bands: str,
        output_dir: Path,
        psf_dir:    Optional[Path],
        invvar_dir: Optional[Path],
        model_dir:  Optional[Path],
        resid_dir:  Optional[Path],
    ) -> dict:
        name = source["name"]
        ra   = source["ra"]
        dec  = source["dec"]
        npix = int(source["size_pixels"])

        result = {"name": name, "ra": ra, "dec": dec, "size_pixels": npix,
                  "band": bands}

        ok, etype, emsg = self.download_cutout(name, ra, dec, npix, bands, output_dir)
        result["image_success"]    = ok
        result["image_error_type"] = etype
        result["image_error_msg"]  = emsg

        if psf_dir is not None:
            ok, etype, emsg = self.download_psf(name, ra, dec, bands, psf_dir)
            result["psf_success"]    = ok
            result["psf_error_type"] = etype
            result["psf_error_msg"]  = emsg

        if invvar_dir is not None:
            ok, etype, emsg = self.download_cutout(
                name, ra, dec, npix, bands, invvar_dir,
                layer_suffix="", filename_suffix="_invvar",
            )
            result["invvar_success"]    = ok
            result["invvar_error_type"] = etype
            result["invvar_error_msg"]  = emsg

        if model_dir is not None:
            ok, etype, emsg = self.download_cutout(
                name, ra, dec, npix, bands, model_dir,
                layer_suffix="-model", filename_suffix="_model",
            )
            result["model_success"]    = ok
            result["model_error_type"] = etype
            result["model_error_msg"]  = emsg

        if resid_dir is not None:
            ok, etype, emsg = self.download_cutout(
                name, ra, dec, npix, bands, resid_dir,
                layer_suffix="-resid", filename_suffix="_resid",
            )
            result["resid_success"]    = ok
            result["resid_error_type"] = etype
            result["resid_error_msg"]  = emsg

        return result


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def load_catalog(path: str) -> pd.DataFrame:
    with open(path) as fh:
        first_line = fh.readline()
    if first_line.startswith("#"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, comment="#")
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    return df


def get_source_list(
    df: pd.DataFrame,
    pixscale: float,
    auto_size: bool,
    size_pixels: int,
    size_factor: float,
    min_size: int,
    use_source_size: bool = False,
    source_size_factor: float = 3.0,
) -> List[dict]:
    ra_col  = _find_col(df, ["RA",  "ra",   "Ra"])
    dec_col = _find_col(df, ["DEC", "dec",  "Dec", "DE"])
    id_col  = _find_col(df, ["ID",  "id",   "Name", "name", "IAUNAME"])

    if ra_col is None or dec_col is None:
        raise ValueError("Catalog must have RA and DEC columns.")

    # --use-source-size: validate column exists up front; values are in arcsec
    src_size_col = None
    if use_source_size:
        src_size_col = _find_col(df, ["source_size", "SOURCE_SIZE", "Source_Size"])
        if src_size_col is None:
            raise ValueError(
                "--use-source-size requested but catalog has no 'source_size' column "
                "(in arcsec). Add the column or omit --use-source-size."
            )

    # --auto-size: detect which size column is available
    d25_col = _find_col(df, ["D25", "d25"])
    r50_col = _find_col(df, ["R50", "r50"])
    r90_col = _find_col(df, ["R90", "r90"])
    rad_col = _find_col(df, ["Rad", "rad", "RAD"])

    if auto_size:
        size_col_label = None
        if d25_col:
            size_col_label = f"D25 ({d25_col})"
        elif r50_col:
            size_col_label = f"R50 ({r50_col})"
        elif r90_col:
            size_col_label = f"R90 ({r90_col})"
        elif rad_col:
            size_col_label = f"Rad ({rad_col})"
        else:
            print("  [!] --auto-size: no D25/R50/R90/Rad column found; falling back to --size.")
            auto_size = False
        if size_col_label:
            print(f"  [auto-size] Using column: {size_col_label}")

    sources = []
    for idx, row in df.iterrows():
        name = str(row[id_col]) if id_col else f"src_{idx:06d}"
        ra   = float(row[ra_col])
        dec  = float(row[dec_col])

        if use_source_size:
            npix = max(int(float(row[src_size_col]) * source_size_factor / pixscale), min_size)

        elif auto_size:
            val = None
            if d25_col:
                d25 = float(row[d25_col])
                if d25 > 0:
                    # D25 in arcmin → diameter in arcsec → radius in pixels
                    val = int(((d25 * 360) / 2.0) / pixscale * size_factor)
            elif r50_col:
                r50 = float(row[r50_col])
                if r50 > 0 and not np.isnan(r50):
                    val = int((r50 * 4 * 2) / pixscale * size_factor)
            elif r90_col:
                r90 = float(row[r90_col])
                if r90 > 0 and not np.isnan(r90):
                    val = int((r90 * 3) / pixscale * size_factor)
            elif rad_col:
                rad = float(row[rad_col])
                if rad > 0 and not np.isnan(rad):
                    val = int(rad / pixscale * size_factor)
            npix = max(val, min_size) if val is not None else size_pixels

        else:
            npix = size_pixels

        sources.append({"name": name, "ra": ra, "dec": dec, "size_pixels": npix})
    return sources


# ---------------------------------------------------------------------------
# Failure logs
# ---------------------------------------------------------------------------

def load_failure_logs(output_dir: Path) -> Tuple[set, set]:
    """Return (not_in_survey_ids, server_error_ids) from existing log files."""
    not_in_survey_log = output_dir / "failed_not_in_survey.txt"
    server_error_log  = output_dir / "failed_server_errors.txt"

    not_in_survey_ids: set = set()
    server_error_ids:  set = set()

    for log_path, target in [(not_in_survey_log, not_in_survey_ids),
                              (server_error_log,  server_error_ids)]:
        if log_path.exists():
            with open(log_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        target.add(line.split('\t')[0])

    return not_in_survey_ids, server_error_ids


def save_failure_logs(results: list, output_dir: Path):
    """Write separate log files for not-in-survey and server-error failures."""
    not_in_survey_log = output_dir / "failed_not_in_survey.txt"
    server_error_log  = output_dir / "failed_server_errors.txt"

    _PRODUCTS = [
        ("image_success",  "image_error_type",  "image_error_msg",  "image"),
        ("psf_success",    "psf_error_type",    "psf_error_msg",    "psf"),
        ("invvar_success", "invvar_error_type", "invvar_error_msg", "invvar"),
        ("model_success",  "model_error_type",  "model_error_msg",  "model"),
        ("resid_success",  "resid_error_type",  "resid_error_msg",  "resid"),
    ]

    not_in_survey: List[str] = []
    server_errors: List[str] = []

    for r in results:
        for sk, ek, mk, label in _PRODUCTS:
            if sk not in r or r[sk]:
                continue
            entry = (f"{r['name']}\t{r['ra']:.6f}\t{r['dec']:.6f}"
                     f"\t{r.get('band', '')}\t{label}\t{r.get(mk, '')}")
            if r.get(ek) == 'not_in_survey':
                not_in_survey.append(entry)
            else:
                server_errors.append(entry)

    if not_in_survey:
        with open(not_in_survey_log, 'w') as fh:
            fh.write("# Sources not in survey coverage - DO NOT RETRY\n")
            fh.write("# SOURCE_ID\tRA\tDEC\tBAND\tPRODUCT\tERROR\n")
            fh.writelines(e + '\n' for e in not_in_survey)
        print(f"  Saved {len(not_in_survey)} 'not in survey' entries → {not_in_survey_log}")

    if server_errors:
        with open(server_error_log, 'w') as fh:
            fh.write("# Server errors - CAN RETRY\n")
            fh.write("# SOURCE_ID\tRA\tDEC\tBAND\tPRODUCT\tERROR\n")
            fh.writelines(e + '\n' for e in server_errors)
        print(f"  Saved {len(server_errors)} server error entries → {server_error_log}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list, products: List[str]):
    total = len(results)
    labels = {
        "image": "Images", "psf": "PSF models",
        "invvar": "Inv-var maps", "model": "Model images", "resid": "Residuals",
    }
    print(f"\n{'='*60}")
    print("  Legacy Survey Download Summary")
    print(f"{'='*60}")
    print(f"  Cutouts processed : {total}   (source x band)")
    for p in products:
        sk = f"{p}_success"
        n = sum(1 for r in results if r.get(sk, False))
        pct = 100 * n / total if total else 0.0
        print(f"  {labels.get(p, p):<20s}: {n}/{total}  ({pct:.1f}%)")
        n_nis = sum(1 for r in results
                    if not r.get(sk, True) and r.get(f"{p}_error_type") == "not_in_survey")
        n_srv = sum(1 for r in results
                    if not r.get(sk, True) and r.get(f"{p}_error_type") == "server_error")
        if n_nis:
            print(f"    Not in survey  : {n_nis}")
        if n_srv:
            print(f"    Server errors  : {n_srv}  (retryable with --retry)")
    print(f"{'='*60}\n")


def save_summary_csv(results: list, output_dir: Path):
    rows = []
    for r in results:
        row = {"name": r["name"], "ra": r["ra"], "dec": r["dec"],
               "band": r.get("band", ""), "size_pixels": r["size_pixels"]}
        for p in ("image", "psf", "invvar", "model", "resid"):
            sk = f"{p}_success"
            if sk in r:
                row[f"{p}_ok"]  = r[sk]
                row[f"{p}_err"] = r.get(f"{p}_error_msg", "")
        rows.append(row)
    csv_path = output_dir / "download_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Summary CSV: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Legacy Survey cutout downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("catalog",    nargs="?", default=None,
                        help="Input CSV catalog with RA, DEC columns "
                             "(omit when using --source-name)")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory for output FITS cutouts")

    # Single-source mode
    one = parser.add_argument_group("Single-source mode")
    one.add_argument("--source-name", default=None, metavar="NAME",
                     help="Download a single source instead of a catalog. Output "
                          "files are named {NAME}_{band}.fits, one file per band.")
    one.add_argument("--ra", default=None, metavar="RA",
                     help="RA of the source (decimal degrees or sexagesimal hours, "
                          "e.g. 01:34:01.4). With --dec this skips name resolution.")
    one.add_argument("--dec", default=None, metavar="DEC",
                     help="Dec of the source (decimal degrees or sexagesimal, "
                          "e.g. +20:35:10)")

    # Data release
    dr = parser.add_argument_group("Data release")
    dr.add_argument("--dr", default="9", metavar="DR",
                    help="Legacy Survey data release (default: 9)")

    # Bands
    b = parser.add_argument_group("Bands")
    b.add_argument("--bands", default="g", metavar="BANDS",
                   help="Bands to download (default: g). Examples: 'grz', 'r', 'iz'. "
                        "'ALL' takes every band of the data release and writes one "
                        "FITS file per band. In --source-name mode a multi-band "
                        "string is always split into one file per band.")

    # Pixel scale
    px = parser.add_argument_group("Pixel scale")
    px.add_argument("--pixscale", type=float, default=LS_PIXSCALE_NATIVE,
                    help=f"Output pixel scale in arcsec/pixel (default: {LS_PIXSCALE_NATIVE})")

    # Cutout size — all in pixels
    s = parser.add_argument_group("Cutout size (pixels)")
    s.add_argument("--size", type=int, default=int(128*6),
                   help="Fixed cutout size in pixels (default: 768)")
    s.add_argument("--auto-size", action="store_true",
                   help="Derive size from catalog columns D25 (arcmin), R50/R90/Rad (arcsec)")
    s.add_argument("--size-factor", type=float, default=1.0,
                   help="Extra multiplier applied on top of the auto-size formula (default: 1.0)")
    s.add_argument("--min-size", type=int, default=128,
                   help="Minimum cutout size in pixels for auto-size and --use-source-size (default: 128)")
    s.add_argument("--use-source-size", action="store_true",
                   help="Derive cutout size from 'source_size' column in catalog (arcsec)")
    s.add_argument("--source-size-factor", type=float, default=3.0,
                   help="Multiply source_size by this factor when --use-source-size is set "
                        "(default: 3.0)")

    # Products
    p = parser.add_argument_group("Products")
    p.add_argument("--psf",    action="store_true",
                   help="Download PSF models (coadd-psf endpoint, per-band FITS)")
    p.add_argument("--invvar", action="store_true",
                   help="Download inverse variance maps")
    p.add_argument("--model",  action="store_true",
                   help="Download Tractor model images")
    p.add_argument("--resid",  action="store_true",
                   help="Download residual images (data - model)")

    # Download behaviour
    d = parser.add_argument_group("Download behaviour")
    d.add_argument("--n-jobs",      type=int,   default=1,
                   help="Parallel threads (default: 4)")
    d.add_argument("--timeout",     type=int,   default=30,
                   help="Request timeout in seconds (default: 30)")
    d.add_argument("--max-retries", type=int,   default=1,
                   help="Retries per file (default: 1)")
    d.add_argument("--retry",       action="store_true",
                   help="Retry mode: skip sources logged as not-in-survey, "
                        "only process those with server errors or missing files")
    d.add_argument("--no-summary-csv", action="store_true",
                   help="Do not write download_summary.csv")

    args = parser.parse_args()

    # Resolve the two operating modes: catalog vs single source. In
    # single-source mode the only positional is the output directory, which
    # argparse parks in `catalog`.
    if args.source_name is not None:
        if args.output_dir is None:
            args.output_dir, args.catalog = args.catalog, None
        if args.catalog is not None:
            parser.error("With --source-name, pass only the output directory "
                         "as a positional argument (no catalog).")
        if args.output_dir is None:
            parser.error("An output directory is required.")
    else:
        if args.ra is not None or args.dec is not None:
            parser.error("--ra/--dec require --source-name.")
        if args.catalog is None or args.output_dir is None:
            parser.error("catalog and output_dir are required unless "
                         "--source-name is used.")

    return args


def main():
    args = parse_args()
    single_source = args.source_name is not None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psf_dir    = output_dir / "psf"    if args.psf    else None
    invvar_dir = output_dir / "invvar" if args.invvar else None
    model_dir  = output_dir / "model"  if args.model  else None
    resid_dir  = output_dir / "resid"  if args.resid  else None
    for d in [psf_dir, invvar_dir, model_dir, resid_dir]:
        if d is not None:
            d.mkdir(exist_ok=True)

    if single_source:
        from source_resolver import resolve_target

        print(f"Single-source mode: {args.source_name}")
        try:
            ra_deg, dec_deg = resolve_target(
                args.source_name, ra=args.ra, dec=args.dec, verbose=True
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        if args.auto_size or args.use_source_size:
            print("  [!] --auto-size / --use-source-size need catalog columns; "
                  f"using --size {args.size} px instead.")
        sources = [{"name": args.source_name, "ra": ra_deg, "dec": dec_deg,
                    "size_pixels": args.size}]
        print()
    else:
        print(f"Loading catalog: {args.catalog}")
        try:
            df = load_catalog(args.catalog)
        except Exception as exc:
            print(f"ERROR loading catalog: {exc}")
            sys.exit(1)
        print(f"  {len(df)} sources found.\n")

        try:
            sources = get_source_list(
                df,
                pixscale=args.pixscale,
                auto_size=args.auto_size,
                size_pixels=args.size,
                size_factor=args.size_factor,
                min_size=args.min_size,
                use_source_size=args.use_source_size,
                source_size_factor=args.source_size_factor,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    # One entry per output FITS file
    band_list = expand_bands(args.bands, args.dr, split=single_source)

    downloader = LegacySurveyDownloader(
        data_release=args.dr,
        pixscale=args.pixscale,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    # Retry-mode filtering
    not_in_survey_ids, server_error_ids = load_failure_logs(output_dir)
    n_jobs = args.n_jobs
    if args.retry:
        before = len(sources)
        filtered = []
        for src in sources:
            name = src["name"]
            if name in not_in_survey_ids:
                continue
            img_ok = all((output_dir / f"{name}_{b}.fits").exists()
                         for b in band_list)
            psf_ok = psf_dir is None or all(
                (psf_dir / f"psf_{name}_{b}.fits").exists() for b in band_list
            )
            if not img_ok or not psf_ok or name in server_error_ids:
                filtered.append(src)
        sources = filtered
        n_jobs = min(n_jobs, 2)
        print(f"  Retry mode: {before - len(sources)} sources skipped "
              f"({len(not_in_survey_ids)} not-in-survey), "
              f"{len(sources)} to process.\n")

    products = ["image"]
    if args.psf:
        products.append("psf")
    if args.invvar:
        products.append("invvar")
    if args.model:
        products.append("model")
    if args.resid:
        products.append("resid")

    print("Configuration:")
    print(f"  Data release   : DR{args.dr}")
    print(f"  Bands          : {', '.join(band_list)}"
          + ("  (one FITS per band)" if len(band_list) > 1 else ""))
    print(f"  Pixel scale    : {args.pixscale} arcsec/pix")
    if single_source:
        print(f"  Source         : {sources[0]['name']}  "
              f"RA={sources[0]['ra']:.6f}  Dec={sources[0]['dec']:.6f}")
    if args.use_source_size and not single_source:
        print(f"  Cutout size    : source_size [arcsec] × {args.source_size_factor} / {args.pixscale} arcsec/pix  "
              f"(per-source, min {args.min_size} pix)")
    elif args.auto_size and not single_source:
        print(f"  Cutout size    : auto from catalog x {args.size_factor}  "
              f"(min {args.min_size} pix)")
    else:
        print(f"  Cutout size    : {args.size} x {args.size} pix  "
              f"(~{args.size * args.pixscale:.1f} arcsec)")
    print(f"  Products       : {', '.join(products)}")
    if args.psf and abs(args.pixscale - LS_PIXSCALE_NATIVE) > 1e-4:
        print(f"  [!] PSF WARNING: coadd-psf endpoint returns PSF at native "
              f"{LS_PIXSCALE_NATIVE} arcsec/pix, not {args.pixscale} arcsec/pix. "
              f"Resample before using for convolution/fitting.")
    print(f"  Parallel jobs  : {n_jobs}" +
          ("  (capped for retry mode)" if args.retry else ""))
    print(f"  Output dir     : {output_dir}\n")

    jobs = [(src, band) for src in sources for band in band_list]
    all_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(downloader.process_source)(
            src, band, output_dir, psf_dir, invvar_dir, model_dir, resid_dir,
        )
        for src, band in tqdm(jobs, desc="Downloading", unit="cutout")
    )

    save_failure_logs(all_results, output_dir)
    print_summary(all_results, products)

    if not args.no_summary_csv:
        save_summary_csv(all_results, output_dir)


# ---------------------------------------------------------------------------
# Notebook-compatible convenience wrapper
# ---------------------------------------------------------------------------

def download_legacy_survey_data(
    catalog_path: str,
    output_dir: str,
    data_release: str = "9",
    bands: str = "g",
    pixscale: float = LS_PIXSCALE_NATIVE,
    fixed_size_pixels: Optional[int] = None,
    auto_size: bool = False,
    size_factor: float = 1.0,
    min_size: int = 128,
    use_source_size: bool = False,
    source_size_factor: float = 3.0,
    n_jobs: int = 1,
    timeout: int = 30,
    max_retries: int = 1,
    retry_mode: bool = False,
    psf: bool = True,
    invvar: bool = False,
    model: bool = False,
    resid: bool = False,
) -> list:
    """
    Convenience wrapper for notebook use.

    Parameters
    ----------
    catalog_path : str
        Path to CSV catalog with RA, DEC columns.
    output_dir : str
        Output directory for downloaded files.
    data_release : str
        DR version (default: "9").
    bands : str
        Band string, e.g. "g", "grz" (default: "g").
    pixscale : float
        Pixel scale in arcsec/pixel (default: 0.262).
    fixed_size_pixels : int, optional
        Fixed cutout size in pixels; overrides auto-size/source-size.
    auto_size : bool
        Derive size from D25/R50/R90/Rad catalog columns.
    size_factor : float
        Extra multiplier for auto-size result (default: 1.0).
    min_size : int
        Minimum cutout size in pixels (default: 128).
    use_source_size : bool
        Derive size from 'source_size' catalog column (arcsec) x source_size_factor.
    source_size_factor : float
        Factor applied to source_size column (default: 3.0).
    n_jobs : int
        Parallel download threads (default: 4).
    timeout : int
        HTTP request timeout in seconds (default: 30).
    max_retries : int
        Retry attempts per file (default: 1).
    retry_mode : bool
        Skip not-in-survey sources; only retry server errors (default: False).
    psf : bool
        Download PSF models (default: True).
    invvar : bool
        Download inverse variance maps (default: False).
    model : bool
        Download Tractor model images (default: False).
    resid : bool
        Download residual images (default: False).

    Returns
    -------
    list of dict
        Per-source download results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    psf_dir    = output_path / "psf"    if psf    else None
    invvar_dir = output_path / "invvar" if invvar else None
    model_dir  = output_path / "model"  if model  else None
    resid_dir  = output_path / "resid"  if resid  else None
    for d in [psf_dir, invvar_dir, model_dir, resid_dir]:
        if d is not None:
            d.mkdir(exist_ok=True)

    print(f"Loading catalog: {catalog_path}")
    df = load_catalog(catalog_path)
    print(f"  {len(df)} sources found.")

    size_px    = fixed_size_pixels if fixed_size_pixels is not None else 640
    _auto_size = auto_size and fixed_size_pixels is None
    _src_size  = use_source_size and fixed_size_pixels is None

    sources = get_source_list(
        df,
        pixscale=pixscale,
        auto_size=_auto_size,
        size_pixels=size_px,
        size_factor=size_factor,
        min_size=min_size,
        use_source_size=_src_size,
        source_size_factor=source_size_factor,
    )

    downloader = LegacySurveyDownloader(
        data_release=data_release,
        pixscale=pixscale,
        timeout=timeout,
        max_retries=max_retries,
    )

    not_in_survey_ids, server_error_ids = load_failure_logs(output_path)
    if retry_mode:
        filtered = []
        for src in sources:
            name = src["name"]
            if name in not_in_survey_ids:
                continue
            img_ok = (output_path / f"{name}_{bands}.fits").exists()
            psf_ok = psf_dir is None or (psf_dir / f"psf_{name}_{bands}.fits").exists()
            if not img_ok or not psf_ok or name in server_error_ids:
                filtered.append(src)
        print(f"\n  Retry mode: {len(sources) - len(filtered)} skipped, "
              f"{len(filtered)} to process.")
        sources = filtered

    products = ["image"]
    if psf:
        products.append("psf")
    if invvar:
        products.append("invvar")
    if model:
        products.append("model")
    if resid:
        products.append("resid")

    size_label = (f"{size_px} pix (fixed)" if fixed_size_pixels
                  else "auto / source-size")
    print("\nDownload Configuration:")
    print(f"  Data Release   : DR{data_release}")
    print(f"  Bands          : {bands}")
    print(f"  Pixel Scale    : {pixscale} arcsec/pix")
    print(f"  Cutout size    : {size_label}")
    print(f"  Products       : {', '.join(products)}")
    print(f"  Parallel jobs  : {n_jobs}\n")

    all_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(downloader.process_source)(
            src, bands, output_path, psf_dir, invvar_dir, model_dir, resid_dir,
        )
        for src in tqdm(sources, desc="Downloading", unit="source")
    )

    save_failure_logs(all_results, output_path)
    print_summary(all_results, products)
    save_summary_csv(all_results, output_path)

    return all_results


if __name__ == "__main__":
    main()
