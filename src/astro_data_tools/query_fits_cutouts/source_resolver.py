#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source name / coordinate resolution helpers.
Date: August 2026
Author: Geferson Lucatelli

Shared by the single-source (--source-name) mode of the survey cutout

Resolution order for a source name (first hit wins):
    1. SIMBAD TAP        - fast (~0.2 s), exact identifier match
    2. Sesame / CDS      - astropy's SkyCoord.from_name (SIMBAD + NED + VizieR)
    3. NED               - astroquery, resolves IRAS FSC names the others miss

Each service is tried over the name variants produced by
normalize_source_name(), which repairs the usual catalog spacing issues
('IRASF17132+5313' -> 'IRAS F17132+5313', 'Mrk331' -> 'Mrk 331', ...).

Install (only needed for name resolution, not for --ra/--dec):
    pip install astroquery
"""

import re
from typing import List, Optional, Tuple

__all__ = [
    "normalize_source_name",
    "parse_coordinate",
    "resolve_source_coordinates",
    "resolve_target",
]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalize_source_name(name: str) -> List[str]:
    """
    Generate candidate name variants for a source, handling common
    catalog formatting issues that cause NED/SIMBAD lookup failures.

    Returns a list of candidates in priority order (original first).

    Catalogs handled
    ----------------
    IRAS/IRASF  : 'IRASF17132+5313' -> 'IRAS F17132+5313', 'IRAS 17132+5313'
    Zwicky      : 'IIIZw035'        -> 'III Zw 035', 'III Zw 35'
    VV          : 'VV250'           -> 'VV 250', 'VV 250a', 'VV 250b'
    MCG         : 'MCG+02-01-051'   -> 'MCG +02-01-051', 'MCG 02-01-051'
    Generic     : 'NGC5256'         -> 'NGC 5256'  (alpha + digits)
    """
    name = str(name).strip()
    candidates = [name]

    def add(c):
        if c not in candidates:
            candidates.append(c)

    # -- IRAS Faint Source Catalog: IRASF[hhmm(m)+ddmm] -----------------------
    m = re.match(r'^IRASF(\d{4,5}[+-]\d{4})$', name, re.IGNORECASE)
    if m:
        coords = m.group(1)
        add(f"IRAS F{coords}")
        add(f"IRAS f{coords}")
        add(f"IRAS {coords}")
        return candidates

    # -- Plain IRAS: IRAS[hhmm(m)+ddmm] ---------------------------------------
    m = re.match(r'^IRAS(\d{4,5}[+-]\d{4})$', name, re.IGNORECASE)
    if m:
        coords = m.group(1)
        add(f"IRAS {coords}")
        add(f"IRAS F{coords}")
        return candidates

    # -- Zwicky: [I|II|III|IV]Zw[NNN] -----------------------------------------
    m = re.match(r'^(I{1,3}V?|IV)(Zw)(\d+)$', name, re.IGNORECASE)
    if m:
        roman, zw, num = m.group(1).upper(), 'Zw', m.group(3)
        add(f"{roman} {zw} {num.zfill(3)}")
        add(f"{roman} {zw} {int(num)}")
        return candidates

    # -- VV catalog: VV[NNN] ---------------------------------------------------
    m = re.match(r'^VV(\d+)([ab]?)$', name, re.IGNORECASE)
    if m:
        num, suffix = m.group(1), m.group(2).lower()
        add(f"VV {num}")
        if not suffix:
            # Try component suffixes - NED/SIMBAD often require them
            add(f"VV {num}a")
            add(f"VV {num}b")
        else:
            add(f"VV {num}{suffix}")
        return candidates

    # -- MCG: MCG[+/-][ll]-[gg]-[nnn] -----------------------------------------
    m = re.match(r'^MCG([+-]?\d{2}-\d{2}-\d{3})$', name, re.IGNORECASE)
    if m:
        coords = m.group(1)
        sign = '+' if not coords.startswith('-') else ''
        add(f"MCG {sign}{coords}")
        add(f"MCG {coords.lstrip('+-')}")
        return candidates

    # -- Generic: insert space between alpha prefix and digits -----------------
    # e.g. 'UGC12150' -> 'UGC 12150', 'Mrk331' -> 'Mrk 331'
    m = re.match(r'^([A-Za-z]+)(\d+.*)$', name)
    if m:
        add(f"{m.group(1)} {m.group(2)}")

    return candidates


# ---------------------------------------------------------------------------
# Explicit coordinate parsing
# ---------------------------------------------------------------------------

def parse_coordinate(ra_value, dec_value) -> Tuple[float, float]:
    """
    Convert a user-supplied RA/Dec pair to decimal degrees.

    Accepts decimal degrees ('202.4696', '47.1953') or sexagesimal
    ('13:29:52.7', '+47:11:43'  /  '13h29m52.7s', '+47d11m43s').
    Sexagesimal RA is interpreted as hours, sexagesimal Dec as degrees.

    Raises
    ------
    ValueError
        If the pair cannot be interpreted as a sky position.
    """
    try:
        return float(ra_value), float(dec_value)
    except (TypeError, ValueError):
        pass

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    try:
        coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg))
    except Exception as exc:
        raise ValueError(
            f"Could not parse coordinates RA='{ra_value}' DEC='{dec_value}': {exc}"
        )
    return float(coord.ra.deg), float(coord.dec.deg)


# ---------------------------------------------------------------------------
# Name -> coordinate resolution
# ---------------------------------------------------------------------------

def _resolve_simbad_tap(candidates: List[str], verbose: bool) -> Optional[Tuple[float, float]]:
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        if verbose:
            print("[SIMBAD]    astroquery not installed - skipping")
        return None

    for name in candidates:
        try:
            escaped = name.replace("'", "''")
            result = Simbad.query_tap(
                f"SELECT main_id, ra, dec "
                f"FROM basic "
                f"JOIN ident ON ident.oidref = basic.oid "
                f"WHERE ident.id = '{escaped}'"
            )
            if result is None or len(result) == 0:
                continue
            ra, dec = result['ra'][0], result['dec'][0]
            if ra is None or dec is None:
                continue
            if (hasattr(ra, 'mask') and ra.mask) or (hasattr(dec, 'mask') and dec.mask):
                continue
            if verbose:
                print(f"[SIMBAD]    RA={float(ra):.6f}  Dec={float(dec):.6f}"
                      + (f"  (as '{name}')" if name != candidates[0] else ""))
            return float(ra), float(dec)
        except Exception:
            continue
    return None


def _resolve_sesame(candidates: List[str], verbose: bool) -> Optional[Tuple[float, float]]:
    from astropy.coordinates import SkyCoord

    for name in candidates:
        try:
            coord = SkyCoord.from_name(name)
        except Exception:
            continue
        if verbose:
            print(f"[Sesame]    RA={coord.ra.deg:.6f}  Dec={coord.dec.deg:.6f}"
                  + (f"  (as '{name}')" if name != candidates[0] else ""))
        return float(coord.ra.deg), float(coord.dec.deg)
    return None


def _resolve_ned(candidates: List[str], verbose: bool,
                 timeout: int = 20) -> Optional[Tuple[float, float]]:
    try:
        from astroquery.ipac.ned import Ned
    except ImportError:
        if verbose:
            print("[NED]       astroquery not installed - skipping")
        return None

    Ned.TIMEOUT = timeout
    for name in candidates:
        try:
            table = Ned.query_object(name)
            if table is None or len(table) == 0:
                continue
            ra, dec = float(table['RA'][0]), float(table['DEC'][0])
            if verbose:
                print(f"[NED]       RA={ra:.6f}  Dec={dec:.6f}"
                      + (f"  (as '{name}')" if name != candidates[0] else ""))
            return ra, dec
        except Exception:
            continue
    return None


def resolve_source_coordinates(source_name: str,
                               verbose: bool = True,
                               ned_timeout: int = 20) -> Optional[Tuple[float, float]]:
    """
    Resolve a source name to (RA, Dec) in decimal degrees (ICRS).

    Query order: SIMBAD TAP -> Sesame/CDS -> NED.

    Parameters
    ----------
    source_name : str
        Source name, e.g. 'Mrk331', 'VV705', 'IRASF17132+5313'.
    verbose : bool
        Print which service resolved the name.
    ned_timeout : int
        Per-attempt read timeout for the NED fallback, in seconds.

    Returns
    -------
    (ra, dec) : tuple of float, or None
        Decimal degrees, or None if no service could resolve the name.
    """
    candidates = normalize_source_name(source_name)
    if verbose:
        print(f"Resolving '{source_name}' (variants tried: {candidates})")

    for resolver in (_resolve_simbad_tap, _resolve_sesame):
        coords = resolver(candidates, verbose)
        if coords is not None:
            return coords

    coords = _resolve_ned(candidates, verbose, timeout=ned_timeout)
    if coords is not None:
        return coords

    if verbose:
        print(f"[!] Could not resolve '{source_name}'. "
              f"Pass the position explicitly with --ra / --dec.")
    return None


def resolve_target(source_name: str,
                   ra=None,
                   dec=None,
                   verbose: bool = True) -> Tuple[float, float]:
    """
    Return (RA, Dec) in degrees for a single target.

    If both *ra* and *dec* are given they are parsed and used directly (no
    network query).  Otherwise *source_name* is resolved via
    resolve_source_coordinates().

    Raises
    ------
    ValueError
        If only one of ra/dec is given, if the pair cannot be parsed, or if
        the name cannot be resolved by any service.
    """
    if (ra is None) != (dec is None):
        raise ValueError("--ra and --dec must be given together.")

    if ra is not None:
        ra_deg, dec_deg = parse_coordinate(ra, dec)
        if verbose:
            print(f"Using supplied position for '{source_name}': "
                  f"RA={ra_deg:.6f}  Dec={dec_deg:.6f}")
        return ra_deg, dec_deg

    coords = resolve_source_coordinates(source_name, verbose=verbose)
    if coords is None:
        raise ValueError(
            f"Could not resolve coordinates for '{source_name}'. "
            f"Provide them explicitly with --ra and --dec."
        )
    return coords
