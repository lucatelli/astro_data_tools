#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

HSC Survey Data Downloader with Authentication
Date: Original version November 2024; latest update August 2026, v3
Author: Geferson Lucatelli & Claude Code

Downloads cutouts from the HSC (Hyper Suprime-Cam) Survey using the official DAS cutout service.
Supports authentication and multiple data products with clear pixel-based size control.

Usage:
    python get_data_hsc_survey.py <catalog.csv> <output_directory> [options]
    python get_data_hsc_survey.py --source-name <NAME> <output_directory> [options]

Your catalog.csv must have #ID,RA,DEC or ID,RA,DEC columns.

Example (fixed size):
    python get_data_hsc_survey.py sources.csv ./output/ -u user --cutout-size 256 --filter HSC-I

Example (catalog-based with R50):
    python get_data_hsc_survey.py sources.csv ./output/ -u user --catalog-size-column R50 --size-multiplier 6 --min-cutout-size 128 --filter HSC-I

Example (single source, name resolved via SIMBAD/Sesame/NED, all broad bands):
    python get_data_hsc_survey.py --source-name Mrk331 ./output/Mrk331/ -u user --filter ALL --download-psf
    -> ./output/Mrk331/Mrk331_g.fits, Mrk331_r.fits, Mrk331_i.fits, ...

Example (single source at an explicit position, no name resolution):
    python get_data_hsc_survey.py --source-name Mrk331 --ra 23.5058 --dec 20.5862 ./output/Mrk331/ -u user --filter ALL

Data products and where they land:

    image (always)      {output}/{source}_{band}.fits
    --download-mask     {output}/mask/mask_{source}_{band}.fits
    --download-variance {output}/variance/variance_{source}_{band}.fits
    --download-psf      {output}/psf/psf_{source}_{band}.fits

    HSC returns the image, mask and variance planes together in one
    multi-extension response; this script splits them into one file per
    plane so that each requested product is a visible file. If a position
    was already downloaded without a plane, add --overwrite to fetch it -
    the file names do not record which planes were asked for, so an
    existing image file would otherwise satisfy the request.

Cutout type (--type):

    coadd (default)  The stacked survey image. One cutout per band, which
                     is what you want for photometry and morphology.
    warp             Every individual warped exposure overlapping the
                     position. The service answers with a tar archive
                     rather than a cutout, so it is unpacked into
                     warp/{source}_{band}/ - useful for variability or
                     for building your own stack, not for a single image.

Note: You need an HSC account to download data. Register at:
https://hsc-release.mtk.nao.ac.jp/
"""

import argparse
import getpass
import io
import os
import sys
import tarfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
from joblib import Parallel, delayed
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Broad-band filters expanded by --filter ALL
HSC_BROAD_FILTERS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]


def _canonical_hsc_filter(token: str) -> str:
    """
    Map a user filter token to the value the HSC DAS service expects.

    'g' / 'G' / 'hsc-g' -> 'HSC-G';  'nb0816' -> 'NB0816';  anything else is
    passed through untouched.
    """
    t = token.strip()
    if t.upper() in {"G", "R", "I", "Z", "Y"}:
        return f"HSC-{t.upper()}"
    if t.upper().startswith("HSC-"):
        return f"HSC-{t[4:].upper()}"
    if t.upper().startswith("NB"):
        return t.upper()
    return t


def resolve_filters(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Expand the --filter tokens into (service_filter, file_label) pairs.

    'ALL' expands to the five broad bands and labels the output files with the
    short band letter ({source}_g.fits, {source}_r.fits, ...).  An explicitly
    named filter keeps the token as its file label, so existing catalog
    downloads keep the filenames they already have on disk.
    """
    pairs: List[Tuple[str, str]] = []
    for token in tokens:
        if token.strip().upper() == "ALL":
            for filt in HSC_BROAD_FILTERS:
                pairs.append((filt, filt.replace("HSC-", "").lower()))
        else:
            pairs.append((_canonical_hsc_filter(token), token.strip()))

    # De-duplicate on the service filter, keeping the first label seen
    seen = set()
    unique: List[Tuple[str, str]] = []
    for filt, label in pairs:
        if filt not in seen:
            seen.add(filt)
            unique.append((filt, label))
    return unique


class HSCDownloader:
    """
    Downloader for HSC Survey cutouts with authentication support.
    Uses the official HSC Data Archive System (DAS) cutout service.
    """
    
    def __init__(
        self,
        username: str,
        password: str,
        data_release: str = "pdr3",
        rerun: str = "pdr3_wide",
        pixscale: float = 0.168,  # HSC pixel scale in arcsec/pixel
        timeout: int = 15,
        max_retries: int = 1,
    ):
        """
        Initialize the HSC downloader with authentication.
        
        Parameters
        ----------
        username : str
            HSC account username
        password : str
            HSC account password
        data_release : str
            Data release version (e.g., "pdr3", "pdr2")
        rerun : str
            Rerun name (e.g., "pdr3_wide", "pdr3_dud")
        pixscale : float
            HSC pixel scale in arcsec/pixel (0.168 is standard)
        timeout : int
            Request timeout in seconds
        max_retries : int
            Maximum number of retry attempts per download
        """
        self.username = username
        self.password = password
        self.data_release = data_release
        self.rerun = rerun
        self.pixscale = pixscale
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HSC DAS cutout service URL
        self.base_url = f"https://hsc-release.mtk.nao.ac.jp/das_cutout/{data_release}/cgi-bin/cutout"
        
        # Create authenticated session
        self.session = requests.Session()
        self.session.auth = (username, password)
        
        # Verify authentication
        self._verify_authentication()
    
    def _verify_authentication(self):
        """
        Verify that authentication credentials are valid by making a test request.
        """
        print(f"Testing authentication for user: {self.username}")
        
        try:
            # Make a minimal test request
            test_params = {
                'ra': 150.0,
                'dec': 2.0,
                'sw': 0.01,
                'sh': 0.01,
                'type': 'coadd',
                'image': 'on',
                'filter': 'HSC-G',
                'rerun': self.rerun
            }
            
            print(f"Testing URL: {self.base_url}")
            response = self.session.get(self.base_url, params=test_params, timeout=30)
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 401:
                print("ERROR: 401 Unauthorized - Authentication failed")
                print("Please verify your HSC username and password are correct")
                raise ValueError("Authentication failed. Please check your HSC username and password.")
            elif response.status_code == 403:
                print("ERROR: 403 Forbidden - Access denied")
                raise ValueError("Access forbidden. Please check your account permissions.")
            elif response.status_code != 200:
                print(f"Warning: Test request returned status code {response.status_code}")
            else:
                print("+> Authentication successful!")
                
        except requests.Timeout:
            print("Warning: Authentication test timed out, but will proceed anyway.")
        except ValueError:
            raise  # Re-raise authentication errors
        except Exception as e:
            print(f"Warning during authentication test: {e}")
            print("Will proceed anyway, but downloads may fail if credentials are incorrect.")
    
    def pixels_to_degrees_semiwidth(self, size_pixels: float) -> float:
        """
        Convert total cutout size in pixels to semi-width in degrees for HSC API.
        
        Parameters
        ----------
        size_pixels : float
            Total cutout size in pixels (width or height)
            
        Returns
        -------
        size_degrees : float
            Semi-width in degrees for HSC sw/sh parameters
            
        Notes
        -----
        Conversion steps:
        1. Total size in pixels -> semi-width in pixels (divide by 2)
        2. Semi-width in pixels -> semi-width in arcsec (multiply by pixscale)
        3. Semi-width in arcsec -> semi-width in degrees (divide by 3600)
        
        Example: 384 pixels total
          -> 192 pixels semi-width
          -> 192 * 0.168 = 32.256 arcsec semi-width
          -> 32.256 / 3600 = 0.00896 degrees semi-width
          -> HSC will return 0.00896*2*3600/0.168 = 384 pixels total
        """
        size_pixels_semiwidth = size_pixels / 2.0
        size_arcsec_semiwidth = size_pixels_semiwidth * self.pixscale
        size_degrees_semiwidth = size_arcsec_semiwidth / 3600.0
        return size_degrees_semiwidth
    
    def read_catalog(
        self, 
        catalog_path: str,
        fixed_size_pixels: Optional[int] = None,
        min_size_pixels: int = 128,
        size_column: str = "auto",
        size_multiplier: float = 4.0,
        user_specified_multiplier: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read source catalog from CSV file and determine cutout sizes.
        
        Parameters
        ----------
        catalog_path : str
            Path to input CSV catalog
        fixed_size_pixels : int or None
            If specified, use this fixed size in pixels for all cutouts (ignores catalog)
        min_size_pixels : int
            Minimum cutout size in pixels when using catalog-based sizing
        size_column : str
            Which column to use for sizing: 'auto', 'D25', 'R50', 'R90', 'Rad', or 'none'
        size_multiplier : float
            Multiplier to apply to catalog size values (only used if user_specified_multiplier=True)
        user_specified_multiplier : bool
            Whether the user explicitly specified a multiplier via command line
            
        Returns
        -------
        ids : np.ndarray
            Source IDs
        ra : np.ndarray
            Right ascension in degrees
        dec : np.ndarray
            Declination in degrees
        size_degrees : np.ndarray
            Cutout semi-width in degrees for HSC API
        """
        print(f"\nReading catalog: {catalog_path}")
        
        try:
            table = Table.read(catalog_path, format='ascii.csv')
            
            # Find RA column (case insensitive)
            ra_col = None
            for col in ['RA', '#RA', 'ra', '#ra']:
                if col in table.colnames:
                    ra_col = col
                    break
            if ra_col is None:
                raise ValueError("No RA column found in catalog")
            
            # Find DEC column
            dec_col = None
            for col in ['DEC', 'Dec', 'dec', 'DE']:
                if col in table.colnames:
                    dec_col = col
                    break
            if dec_col is None:
                raise ValueError("No DEC column found in catalog")
            
            ra = np.array(table[ra_col])
            dec = np.array(table[dec_col])
            
            # Try to find ID column
            id_col = None
            for col in ['ID', '#ID', 'id', '#id', 'ID2']:
                if col in table.colnames:
                    id_col = col
                    break
            
            if id_col is not None:
                ids = np.array(table[id_col], dtype=str)
            else:
                # Generate IDs from RA/DEC
                ids = np.array([f"{r:.6f}_{d:.6f}" for r, d in zip(ra, dec)])
            
            print(f"Found {len(ids)} sources")
            
            # Determine cutout sizes in pixels
            if fixed_size_pixels is not None:
                # Use fixed size for all sources
                print(f"\nSize configuration: FIXED SIZE")
                print(f"  Cutout size: {fixed_size_pixels} pixels")
                print(f"  Angular size: {fixed_size_pixels * self.pixscale:.2f} arcsec")
                size_pixels = np.ones(len(ra)) * fixed_size_pixels
                
            elif size_column == 'none':
                # No catalog column, use minimum size
                print(f"\nSize configuration: MINIMUM SIZE ONLY")
                print(f"  Cutout size: {min_size_pixels} pixels")
                size_pixels = np.ones(len(ra)) * min_size_pixels
                
            else:
                # Use catalog-based sizing
                print(f"\nSize configuration: CATALOG-BASED")
                size_pixels = self._get_catalog_sizes(
                    table, 
                    size_column, 
                    size_multiplier, 
                    min_size_pixels,
                    len(ra),
                    user_specified_multiplier=user_specified_multiplier,
                )
            
            # Convert pixels to degrees (semi-width) for HSC API
            size_degrees = self.pixels_to_degrees_semiwidth(size_pixels)
            
            print(f"\nFinal cutout size distribution:")
            print(f"  Pixels (total):      {size_pixels.min():.0f} - {size_pixels.max():.0f}")
            print(f"  Degrees (semi-width): {size_degrees.min():.6f} - {size_degrees.max():.6f}")
            print(f"  Arcsec (semi-width):  {size_degrees.min()*3600:.2f} - {size_degrees.max()*3600:.2f}")
            print(f"  Arcsec (total):       {size_degrees.min()*2*3600:.2f} - {size_degrees.max()*2*3600:.2f}")
            print(f"\nVerification (converting back to pixels):")
            verify_pixels = (size_degrees * 2 * 3600) / self.pixscale
            print(f"  Expected pixels:      {size_pixels.min():.0f} - {size_pixels.max():.0f}")
            print(f"  Calculated from sw:   {verify_pixels.min():.0f} - {verify_pixels.max():.0f}")
            if np.allclose(verify_pixels, size_pixels):
                print(f"  +> Conversion verified correct")
            
        except Exception as e:
            print(f"Error reading catalog: {e}")
            raise
        
        return ids, ra, dec, size_degrees
    
    def _get_catalog_sizes(
        self,
        table: Table,
        size_column: str,
        size_multiplier: float,
        min_size_pixels: int,
        n_sources: int,
        user_specified_multiplier: bool = False,
    ) -> np.ndarray:
        """
        Extract size information from catalog and convert to pixels.
        
        Parameters
        ----------
        table : Table
            Astropy table with catalog data
        size_column : str
            Which column to use: 'auto', 'D25', 'R50', 'R90', 'Rad'
        size_multiplier : float
            Multiplier for the size column (used if user_specified_multiplier=True)
        min_size_pixels : int
            Minimum size in pixels
        n_sources : int
            Number of sources
        user_specified_multiplier : bool
            Whether the user explicitly specified a multiplier
            
        Returns
        -------
        size_pixels : np.ndarray
            Array of cutout sizes in pixels (total size, not semi-width)
        """
        # Initialize with minimum size
        size_pixels = np.ones(n_sources) * min_size_pixels
        
        # Determine which column to use
        available_columns = []
        for col in ['D25', 'R50', 'R90', 'Rad']:
            if col in table.colnames:
                available_columns.append(col)
        
        if size_column == 'auto':
            # Try columns in order of preference
            for col in ['R50', 'R90', 'D25', 'Rad']:
                if col in available_columns:
                    size_column = col
                    break
            else:
                print(f"  No size columns found. Available columns: {table.colnames}")
                print(f"  Using minimum size: {min_size_pixels} pixels for all sources")
                return size_pixels
        
        if size_column not in table.colnames:
            print(f"  WARNING: Column '{size_column}' not found")
            print(f"  Available size columns: {available_columns}")
            print(f"  Using minimum size: {min_size_pixels} pixels for all sources")
            return size_pixels
        
        # Determine appropriate multiplier based on size metric type
        # Only use defaults if user did not explicitly specify a multiplier
        if not user_specified_multiplier:
            if size_column == 'R50':
                # R50 is half-light radius, use 4x to capture ~90% of light
                actual_multiplier = 4.0
            elif size_column == 'R90':
                # R90 already captures 90% of light, use 2x for safety margin
                actual_multiplier = 2.0
            elif size_column == 'D25':
                # D25 is full diameter at 25 mag/arcsec^2, typically good as-is
                actual_multiplier = 1.0
            elif size_column == 'Rad':
                # Rad is total aperture, use 1.5x for slight margin
                actual_multiplier = 1.5
            else:
                # Unknown type, use provided multiplier
                actual_multiplier = size_multiplier
        else:
            # User explicitly specified multiplier, use it for all types
            actual_multiplier = size_multiplier
        
        print(f"  Size column: '{size_column}'")
        print(f"  Multiplier: {actual_multiplier}x {'(user-specified)' if user_specified_multiplier else '(auto-selected)'}")
        print(f"  Minimum cutout: {min_size_pixels} pixels")
        
        # Get the size values from catalog
        catalog_sizes = np.array(table[size_column])
        
        # Convert to pixels based on column type
        if size_column == 'D25':
            # D25 is typically in arcminutes and represents the full diameter
            # Convert: arcmin -> arcsec -> pixels
            size_pixels_from_catalog = (catalog_sizes * 60.0 / self.pixscale) * actual_multiplier
            print(f"  D25 interpretation: diameter in arcminutes")
            
        elif size_column in ['R50', 'R90']:
            # R50/R90 are typically in arcsec and represent radii
            # Convert: arcsec -> pixels, then apply appropriate multiplier
            size_pixels_from_catalog = (catalog_sizes / self.pixscale) * actual_multiplier
            print(f"  {size_column} interpretation: radius in arcsec")
            
        elif size_column == 'Rad':
            # Rad is typically in arcsec and represents total aperture
            # Convert: arcsec -> pixels, apply multiplier
            size_pixels_from_catalog = (catalog_sizes / self.pixscale) * actual_multiplier
            print(f"  Rad interpretation: total aperture in arcsec")
        
        else:
            # Unknown column type - assume arcsec radius
            print(f"  WARNING: Unknown column type '{size_column}', assuming radius in arcsec")
            size_pixels_from_catalog = (catalog_sizes / self.pixscale) * actual_multiplier
        
        # Apply size_pixels_from_catalog, respecting minimum size
        for k in range(n_sources):
            if (catalog_sizes[k] <= 0 or 
                catalog_sizes[k] is None or 
                np.isnan(catalog_sizes[k])):
                # Invalid value, use minimum
                size_pixels[k] = min_size_pixels
            elif size_pixels_from_catalog[k] < min_size_pixels:
                # Below minimum, use minimum
                size_pixels[k] = min_size_pixels
            else:
                # Valid value, use calculated size
                size_pixels[k] = size_pixels_from_catalog[k]
        
        # Report statistics
        n_min = np.sum(size_pixels == min_size_pixels)
        print(f"  Sources using minimum size: {n_min}/{n_sources}")
        
        return size_pixels
    
    # HSC serves the three planes of a coadd in a fixed order, and the
    # cutout HDUs do not reliably carry an EXTNAME, so the plane a given
    # HDU holds is worked out from which planes were requested.
    PLANE_ORDER = ("image", "mask", "variance")

    def plane_output_path(
        self,
        output_dir: Path,
        plane: str,
        source_id: str,
        label: str,
    ) -> Path:
        """
        Where one plane of a cutout is written.

        The science image keeps the historical name; the mask and variance
        go to their own sub-directories, the same layout the Euclid and
        JWST downloaders use, so that asking for them produces visible
        files rather than extra extensions buried in the image file.
        """
        if plane == "image":
            return output_dir / f"{source_id}_{label}.fits"
        return output_dir / plane / f"{plane}_{source_id}_{label}.fits"

    def download_cutout(
        self,
        source_id: str,
        ra: float,
        dec: float,
        size_degrees: float,
        filter_name: str,
        output_dir: Path,
        cutout_type: str = "coadd",
        download_mask: bool = False,
        download_variance: bool = False,
        file_label: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Download a single HSC cutout with retry logic.

        Parameters
        ----------
        source_id : str
            Source identifier
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        size_degrees : float
            Cutout semi-width in degrees
        filter_name : str
            HSC filter sent to the service (e.g., "HSC-I", "HSC-G", "HSC-R")
        output_dir : Path
            Output directory
        cutout_type : str
            Type of cutout: "coadd" (default) or "warp". A warp request
            returns a tar archive of the individual warped exposures, which
            is unpacked into warp/{source_id}_{label}/ instead of a cutout.
        download_mask : bool
            Whether to download the mask plane
        download_variance : bool
            Whether to download the variance plane
        file_label : str, optional
            Band label used in the output filename. Defaults to filter_name.
        overwrite : bool
            Re-download even when the output files are already present
        verbose : bool
            Print detailed debugging information

        Returns
        -------
        results : dict
            Per-plane success flags, e.g. {'image': True, 'variance': True}
        """
        label = file_label if file_label is not None else filter_name

        # Which planes did the caller ask for? This drives both the request
        # and the skip logic, because a file downloaded earlier without
        # --download-variance must not satisfy a later run that wants it.
        planes = ["image"]
        if download_mask:
            planes.append("mask")
        if download_variance:
            planes.append("variance")

        if cutout_type == "warp":
            return self._download_warp(source_id, ra, dec, size_degrees,
                                       filter_name, output_dir, label,
                                       planes, overwrite, verbose)

        paths = {p: self.plane_output_path(output_dir, p, source_id, label)
                 for p in planes}

        if not overwrite and all(path.exists() for path in paths.values()):
            return {p: True for p in planes}

        # Construct URL parameters
        # HSC uses sw (semi-width) and sh (semi-height) in DEGREES
        params = {
            'ra': ra,
            'dec': dec,
            'sw': size_degrees,
            'sh': size_degrees,
            'type': cutout_type,
            'image': 'on',
            'filter': filter_name,
            'rerun': self.rerun,
        }
        
        # Add optional data products
        if download_mask:
            params['mask'] = 'on'
        if download_variance:
            params['variance'] = 'on'
        
        if verbose:
            # Construct full URL for debugging
            param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{self.base_url}?{param_str}"
            print(f"\nDownloading {source_id}:")
            print(f"  URL: {self.base_url}")
            print(f"  Params: {params}")
            print(f"  Full URL: {full_url}")
            print(f"  Planes requested: {', '.join(planes)}")
        
        # Try download with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                
                if verbose:
                    print(f"  Status code: {response.status_code}")
                    print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
                
                # Check for authentication errors
                if response.status_code == 401:
                    print(f"\nAuthentication failed for {source_id}")
                    print(f"Response: {response.text[:200]}")
                    raise ValueError("Authentication failed. Please check your credentials.")
                
                response.raise_for_status()
                
                # Check if we got an error page instead of FITS data
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type.lower():
                    if verbose:
                        print(f"  ERROR: Received HTML instead of FITS")
                        print(f"  Response: {response.text[:500]}")
                    raise ValueError("Received HTML instead of FITS file - likely an error page")
                
                # Check content size
                content_length = len(response.content)
                if verbose:
                    print(f"  Content length: {content_length} bytes")
                
                if content_length == 0:
                    raise ValueError("Received empty response (0 bytes)")

                written = self._write_planes(response.content, planes, paths,
                                             source_id, ra, dec, filter_name,
                                             label, verbose)
                return written

            except requests.Timeout:
                if verbose:
                    print(f"  Timeout (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt < self.max_retries - 1:
                    continue
                else:
                    print(f"Timeout downloading {source_id} after {self.max_retries} attempts")
                    return {p: False for p in planes}
                    
            except Exception as e:
                if verbose:
                    print(f"  Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2 ** attempt)
                if attempt < self.max_retries - 1:
                    continue
                else:
                    print(f"Error downloading {source_id}: {e}")
                    return {p: False for p in planes}

        return {p: False for p in planes}

    def _write_planes(
        self,
        content: bytes,
        planes: List[str],
        paths: Dict[str, Path],
        source_id: str,
        ra: float,
        dec: float,
        filter_name: str,
        label: str,
        verbose: bool = False,
    ) -> Dict[str, bool]:
        """
        Split an HSC cutout into one file per requested plane.

        A coadd cutout arrives as a single FITS whose primary HDU carries
        only the provenance header and whose data extensions follow the
        fixed order image, mask, variance - restricted to the planes that
        were actually requested. Each is written out separately, with the
        primary header merged in so the provenance is not lost.
        """
        with fits.open(io.BytesIO(content)) as hdul:
            if verbose:
                print(f"  FITS file opened, {len(hdul)} HDU(s)")
                for i, hdu in enumerate(hdul):
                    shape = None if hdu.data is None else hdu.data.shape
                    print(f"  HDU {i}: {type(hdu).__name__}, shape: {shape}, "
                          f"header cards: {len(hdu.header)}")

            primary_header = hdul[0].header
            data_hdus = [h for h in hdul
                         if h.data is not None and h.data.size > 0]
            if not data_hdus:
                raise ValueError("No image data found in any HDU")

            # Prefer the EXTNAME when the service supplies one; otherwise
            # fall back on the documented plane ordering.
            expected = [p for p in self.PLANE_ORDER if p in planes]
            resolved: Dict[str, fits.hdu.base._BaseHDU] = {}
            for position, hdu in enumerate(data_hdus):
                name = str(hdu.header.get('EXTNAME', '')).strip().lower()
                if name in planes:
                    resolved[name] = hdu
                elif position < len(expected):
                    resolved.setdefault(expected[position], hdu)

            results: Dict[str, bool] = {}
            for plane in planes:
                hdu = resolved.get(plane)
                if hdu is None:
                    if verbose:
                        print(f"  [!] {plane}: not present in the response "
                              f"({len(data_hdus)} data HDU(s) returned)")
                    results[plane] = False
                    continue

                header = primary_header.copy()
                header.update(hdu.header)
                header['EXTNAME'] = (plane.upper(), 'HSC cutout plane')
                header['SRCNAME'] = (source_id, 'Requested source name')
                header['SRCRA'] = (ra, 'Requested RA (deg, ICRS)')
                header['SRCDEC'] = (dec, 'Requested Dec (deg, ICRS)')
                header['HSCFILT'] = (filter_name, 'HSC filter')
                header['HSCPROD'] = (plane, 'Plane held by this file')
                header['HSCRERUN'] = (self.rerun, 'HSC rerun')

                path = paths[plane]
                path.parent.mkdir(parents=True, exist_ok=True)
                fits.writeto(str(path), hdu.data, header, overwrite=True,
                             output_verify='silentfix')
                results[plane] = True
                if verbose:
                    print(f"  Saved {plane}: {path}")

            return results

    def _download_warp(
        self,
        source_id: str,
        ra: float,
        dec: float,
        size_degrees: float,
        filter_name: str,
        output_dir: Path,
        label: str,
        planes: List[str],
        overwrite: bool,
        verbose: bool,
    ) -> Dict[str, bool]:
        """
        Fetch the warped single exposures for a position.

        Unlike a coadd request this returns a tar archive holding one FITS
        per contributing exposure, so there is no single cutout to write:
        the archive is unpacked into warp/{source_id}_{label}/ and the
        number of exposures recovered is reported.
        """
        target_dir = output_dir / "warp" / f"{source_id}_{label}"
        if not overwrite and target_dir.is_dir() and any(target_dir.iterdir()):
            return {p: True for p in planes}

        params = {
            'ra': ra,
            'dec': dec,
            'sw': size_degrees,
            'sh': size_degrees,
            'type': 'warp',
            'image': 'on',
            'filter': filter_name,
            'rerun': self.rerun,
        }
        if "mask" in planes:
            params['mask'] = 'on'
        if "variance" in planes:
            params['variance'] = 'on'

        if verbose:
            print(f"\nDownloading warps for {source_id} ({filter_name})")
            print(f"  Params: {params}")

        try:
            response = self.session.get(self.base_url, params=params,
                                        timeout=self.timeout)
            if response.status_code == 401:
                raise ValueError("Authentication failed")
            response.raise_for_status()

            content = response.content
            if not content:
                raise ValueError("Received empty response (0 bytes)")
            if b'ustar' not in content[:1024] and not content[:2] == b'\x1f\x8b':
                # Not a tar: most likely an HTML error page.
                raise ValueError(
                    "warp request did not return a tar archive; the service "
                    "may have no warped exposures for this position")

            target_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(content), mode='r:*') as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                for member in members:
                    # Flatten the archive and refuse absolute or escaping paths.
                    name = Path(member.name).name
                    if not name:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    (target_dir / name).write_bytes(extracted.read())

            if verbose:
                print(f"  Unpacked {len(members)} warped exposure(s) -> {target_dir}")
            return {p: bool(members) for p in planes}

        except Exception as e:
            print(f"Error downloading warps for {source_id}: {e}")
            return {p: False for p in planes}

    def download_psf(
        self,
        source_id: str,
        ra: float,
        dec: float,
        filter_name: str,
        output_dir: Path,
        cutout_type: str = "coadd",
        file_label: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> bool:
        """
        Download PSF model for a given position and filter.

        Parameters
        ----------
        source_id : str
            Source identifier
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        filter_name : str
            HSC filter sent to the service (e.g., "HSC-I", "HSC-G", "HSC-R")
        output_dir : Path
            Output directory
        cutout_type : str
            Type: "coadd" or "warp"
        file_label : str, optional
            Band label used in the output filename. Defaults to filter_name.
        verbose : bool
            Print detailed debugging information

        Returns
        -------
        success : bool
            Whether download succeeded
        """
        # Construct filename: psf_<source>_<label>.fits
        filter_short = filter_name
        label = file_label if file_label is not None else filter_name
        filename = f"psf_{source_id}_{label}.fits"
        output_path = output_dir / filename

        # Skip if already exists
        if output_path.exists() and not overwrite:
            return True

        output_dir.mkdir(parents=True, exist_ok=True)

        # HSC PSF service URL
        psf_url = f"https://hsc-release.mtk.nao.ac.jp/psf/{self.data_release}/cgi/getpsf"
        
        # Construct parameters
        params = {
            'ra': ra,
            'dec': dec,
            'filter': filter_short,
            'rerun': self.rerun,
            'tract': '',
            'patch': '',
            'centered': 'true',
            'type': cutout_type,
        }
        
        if verbose:
            param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{psf_url}?{param_str}"
            print(f"\nDownloading PSF for {source_id} ({filter_name}):")
            print(f"  Full URL: {full_url}")
        
        # Try download with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    psf_url,
                    params=params,
                    timeout=self.timeout
                )
                
                if verbose:
                    print(f"  Status code: {response.status_code}")
                
                if response.status_code == 401:
                    raise ValueError("Authentication failed")
                
                response.raise_for_status()
                
                # Check content
                content_length = len(response.content)
                if verbose:
                    print(f"  Content length: {content_length} bytes")
                
                if content_length == 0:
                    raise ValueError("Received empty PSF response")
                
                # Write to file
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify it's valid FITS
                try:
                    with fits.open(output_path) as hdul:
                        if len(hdul) == 0:
                            raise ValueError("Empty PSF FITS file")
                        if verbose:
                            print(f"  +> PSF downloaded ({len(hdul)} HDU(s))")
                except Exception as e:
                    output_path.unlink()
                    raise ValueError(f"Invalid PSF FITS: {e}")
                
                return True
                
            except requests.Timeout:
                if verbose:
                    print(f"  Timeout (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(15 ** attempt)
                if attempt < self.max_retries - 1:
                    continue
                else:
                    print(f"Timeout downloading PSF for {source_id}")
                    return False
                    
            except Exception as e:
                if verbose:
                    print(f"  Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(15 ** attempt)
                if attempt < self.max_retries - 1:
                    continue
                else:
                    print(f"Error downloading PSF for {source_id}: {e}")
                    return False
        
        return False
    
    def process_source(
        self,
        source_id: str,
        ra: float,
        dec: float,
        size_degrees: float,
        filters: List[Tuple[str, str]],
        output_dir: Path,
        cutout_type: str = "coadd",
        download_mask: bool = False,
        download_variance: bool = False,
        download_psf: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Process a single source: download cutouts for all requested filters.

        Parameters
        ----------
        filters : list of (service_filter, file_label)
            As returned by resolve_filters(). One FITS file is written per
            entry: {source_id}_{file_label}.fits

        Returns
        -------
        results : dict
            Success status keyed by file label, plus '<label>_mask',
            '<label>_variance' and '<label>_psf' entries for the extra
            products that were requested
        """
        results = {}

        for filter_name, file_label in filters:
            planes = self.download_cutout(
                source_id,
                ra,
                dec,
                size_degrees,
                filter_name,
                output_dir,
                cutout_type=cutout_type,
                download_mask=download_mask,
                download_variance=download_variance,
                file_label=file_label,
                overwrite=overwrite,
                verbose=verbose,
            )
            # The science image decides whether the filter counts as done;
            # the mask and variance planes are tracked under their own keys
            # so a missing variance map is visible in the summary.
            success = planes.get("image", False)
            results[file_label] = success
            for plane, ok in planes.items():
                if plane != "image":
                    results[f"{file_label}_{plane}"] = ok

            # Download PSF if requested and cutout was successful
            if download_psf and success:
                psf_success = self.download_psf(
                    source_id,
                    ra,
                    dec,
                    filter_name,
                    output_dir / "psf",
                    cutout_type=cutout_type,
                    file_label=file_label,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                results[f"{file_label}_psf"] = psf_success

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download HSC Survey cutouts with authentication and pixel-based size control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed size for all sources (256x256 pixels)
  python get_data_hsc_survey.py sources.csv ./output/ -u user --cutout-size 256 --filter HSC-I
  
  # Catalog-based sizing using R50 with 6x multiplier, minimum 128 pixels
  python get_data_hsc_survey.py sources.csv ./output/ -u user --catalog-size-column R50 --size-multiplier 6 --min-cutout-size 128 --filter HSC-I
  
  # Multiple filters with variance maps
  python get_data_hsc_survey.py sources.csv ./output/ -u user --cutout-size 512 --filter HSC-G HSC-R HSC-I --download-variance
  
  # Auto-detect size column with custom multiplier
  python get_data_hsc_survey.py sources.csv ./output/ -u user --catalog-size-column auto --size-multiplier 8 --filter HSC-I

  # Single source by name, all broad bands, one FITS per band
  python get_data_hsc_survey.py --source-name Mrk331 ./output/Mrk331/ -u user --filter ALL --download-psf

  # Single source at an explicit position (skips name resolution)
  python get_data_hsc_survey.py --source-name Mrk331 --ra 23.5058 --dec 20.5862 ./output/Mrk331/ -u user --filter ALL

Available filters: HSC-G, HSC-R, HSC-I, HSC-Z, HSC-Y, NB0387, NB0816, NB0921
  (short forms g/r/i/z/y are accepted; ALL expands to the five broad bands)

Size options:
  --cutout-size: Fixed pixel size for all sources (overrides catalog)
  --catalog-size-column: Use catalog column for sizing (D25, R50, R90, Rad, auto, or none)
  --size-multiplier: Multiply catalog values (e.g., 4 means 4*R50 for cutout size)
  --min-cutout-size: Minimum size in pixels when using catalog
        """
    )
    
    parser.add_argument(
        "catalog",
        type=str,
        nargs="?",
        default=None,
        help="Input catalog CSV file with ID,RA,DEC columns. "
             "Omit when using --source-name."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Output directory for downloaded files"
    )

    single = parser.add_argument_group("Single-source mode")
    single.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Download one source instead of a catalog. Output files are named "
             "{source-name}_{band}.fits, one file per band."
    )
    single.add_argument(
        "--ra",
        type=str,
        default=None,
        help="RA of the source (decimal degrees or sexagesimal hours, e.g. 01:34:01.4). "
             "Given together with --dec, this skips name resolution."
    )
    single.add_argument(
        "--dec",
        type=str,
        default=None,
        help="Dec of the source (decimal degrees or sexagesimal, e.g. +20:35:10)"
    )


    parser.add_argument(
        "-u", "--username",
        type=str,
        required=True,
        help="HSC account username (required)"
    )
    
    parser.add_argument(
        "-p", "--password",
        type=str,
        default=None,
        help="HSC account password (will prompt if not provided)"
    )
    
    parser.add_argument(
        "--dr",
        type=str,
        default="pdr3",
        choices=["pdr1", "pdr2", "pdr3"],
        help="Data release version (default: pdr3)"
    )
    
    parser.add_argument(
        "--rerun",
        type=str,
        default="pdr3_wide",
        help="Rerun name (default: pdr3_wide). Common: pdr3_wide, pdr3_dud"
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        nargs='+',
        default=["HSC-G"],
        help="HSC filter(s) to download (default: HSC-G). Example: HSC-G HSC-R HSC-I. "
             "Short forms (g r i z y) are accepted, and ALL expands to the five "
             "broad bands, written as one FITS file per band."
    )
    
    parser.add_argument(
        "--type",
        type=str,
        default="coadd",
        choices=["coadd", "warp"],
        help="Cutout type. 'coadd' (default) is the stacked survey image "
             "and yields one cutout per band. 'warp' returns every "
             "individual warped exposure overlapping the position, as a tar "
             "archive that is unpacked into warp/{source}_{band}/ - not a "
             "single cutout"
    )
    
    parser.add_argument(
        "--cutout-size",
        type=int,
        default=int(128*6),
        help="Fixed cutout size in pixels (total width/height) for all sources. Overrides catalog sizing. Example: 256"
    )
    
    parser.add_argument(
        "--min-cutout-size",
        type=int,
        default=int(128*3),
        help="Minimum cutout size in pixels when using catalog-based sizing (default: 384)"
    )
    
    parser.add_argument(
        "--catalog-size-column",
        type=str,
        default="auto",
        help="Column for catalog-based sizing. Options: 'auto' (tries R50, R90, D25, Rad), 'D25', 'R50', 'R90', 'Rad', 'none'. Default: auto"
    )
    
    parser.add_argument(
        "--size-multiplier",
        type=float,
        default=None,
        help="Multiplier for catalog size. If not specified, uses intelligent defaults: R50=4x, R90=2x, D25=1x, Rad=1.5x. Example: --size-multiplier 6.0 to override"
    )
    
    parser.add_argument(
        "--download-mask",
        action="store_true",
        help="Download the mask plane, written to "
             "mask/mask_{source}_{band}.fits"
    )
    
    parser.add_argument(
        "--download-variance",
        action="store_true",
        help="Download the variance plane, written to "
             "variance/variance_{source}_{band}.fits"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist. Needed when adding "
             "--download-mask or --download-variance to a position that was "
             "downloaded before without them"
    )
    
    parser.add_argument(
        "--download-psf",
        action="store_true",
        help="Download PSF models for each source and filter"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel download jobs (default: 8)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Request timeout in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retry attempts per download (default: 1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debugging information during downloads"
    )
    
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve the two operating modes: catalog vs single source.
    # In single-source mode the only positional argument is the output
    # directory, which argparse parks in `catalog`.
    # ------------------------------------------------------------------
    single_source = args.source_name is not None
    if single_source:
        if args.output_dir is None:
            args.output_dir, args.catalog = args.catalog, None
        if args.catalog is not None:
            parser.error(
                "With --source-name, pass only the output directory as a "
                "positional argument (no catalog)."
            )
        if args.output_dir is None:
            parser.error("An output directory is required.")
    else:
        if args.ra is not None or args.dec is not None:
            parser.error("--ra/--dec require --source-name.")
        if args.catalog is None or args.output_dir is None:
            parser.error("catalog and output_dir are required unless --source-name is used.")

    # Expand/normalise the requested filters into (service_filter, file_label)
    filters = resolve_filters(args.filter)

    # Get password if not provided
    if args.password is None:
        args.password = getpass.getpass(f"HSC password for {args.username}: ")
    
    # Determine if user specified a custom size multiplier
    user_specified_multiplier = args.size_multiplier is not None
    if not user_specified_multiplier:
        # Use a placeholder value that will be replaced by intelligent defaults
        args.size_multiplier = 4.0  # Default fallback, actual values set in _get_catalog_sizes
    
    # Initialize downloader with authentication
    print("="*70)
    print("HSC Survey Cutout Downloader")
    print("="*70)
    downloader = HSCDownloader(
        username=args.username,
        password=args.password,
        data_release=args.dr,
        rerun=args.rerun,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PSF directory if needed
    if args.download_psf:
        (output_dir / "psf").mkdir(exist_ok=True)
    
    if single_source:
        # Single source: resolve the position, then use one fixed cutout size
        from source_resolver import resolve_target

        print(f"\n{'='*70}")
        print(f"Single-source mode: {args.source_name}")
        print(f"{'='*70}")
        try:
            ra_deg, dec_deg = resolve_target(
                args.source_name, ra=args.ra, dec=args.dec, verbose=True
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

        ids = np.array([args.source_name])
        ra = np.array([ra_deg])
        dec = np.array([dec_deg])
        size_degrees = np.array([
            downloader.pixels_to_degrees_semiwidth(args.cutout_size)
        ])
        print(f"  Cutout size: {args.cutout_size} pixels "
              f"({args.cutout_size * downloader.pixscale:.2f} arcsec)")
    else:
        # Read catalog and determine sizes
        ids, ra, dec, size_degrees = downloader.read_catalog(
            args.catalog,
            fixed_size_pixels=args.cutout_size,
            min_size_pixels=args.min_cutout_size,
            size_column=args.catalog_size_column,
            size_multiplier=args.size_multiplier,
            user_specified_multiplier=user_specified_multiplier,
        )

    print(f"\n{'='*70}")
    print("Download Configuration")
    print(f"{'='*70}")
    print(f"  Data Release: {args.dr}")
    print(f"  Rerun: {args.rerun}")
    print(f"  Filters: {', '.join(f for f, _ in filters)}")
    print(f"  Cutout type: {args.type}")
    print(f"  Number of sources: {len(ids)}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"\nData products:")
    print(f"  Image data: Yes")
    print(f"  Mask plane: {args.download_mask}"
          + ("   -> mask/mask_{source}_{band}.fits" if args.download_mask else ""))
    print(f"  Variance plane: {args.download_variance}"
          + ("   -> variance/variance_{source}_{band}.fits"
             if args.download_variance else ""))
    print(f"  PSF models: {args.download_psf}")
    print(f"{'='*70}\n")
    
    # Download in parallel
    print("Starting downloads...")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(downloader.process_source)(
            ids[i],
            ra[i],
            dec[i],
            size_degrees[i],
            filters,
            output_dir,
            cutout_type=args.type,
            download_mask=args.download_mask,
            download_variance=args.download_variance,
            download_psf=args.download_psf,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        for i in tqdm(range(len(ids)), desc="Downloading")
    )
    
    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    
    total = len(results)
    
    # Count successes for each filter
    for filter_name, file_label in filters:
        success = sum(1 for r in results if r.get(file_label, False))
        print(f"  {filter_name}: {success}/{total} ({100*success/total:.1f}%)")

        # The extra planes are counted separately, so that a variance map
        # that never arrived cannot hide behind a successful image.
        for plane, wanted in (("mask", args.download_mask),
                              ("variance", args.download_variance)):
            if not wanted:
                continue
            key = f"{file_label}_{plane}"
            n = sum(1 for r in results if r.get(key, False))
            print(f"  {filter_name} {plane}: {n}/{total} "
                  f"({100*n/total:.1f}%)")

        # PSF statistics if requested
        if args.download_psf:
            psf_key = f"{file_label}_psf"
            psf_success = sum(1 for r in results if r.get(psf_key, False))
            print(f"  {filter_name} PSF: {psf_success}/{total} ({100*psf_success/total:.1f}%)")
    
    # Check for completely failed sources
    failed_ids = [ids[i] for i, r in enumerate(results) 
                  if not any(r.values())]
    
    if failed_ids:
        print(f"\n{len(failed_ids)} sources failed completely:")
        for source_id in failed_ids[:10]:
            print(f"  - {source_id}")
        if len(failed_ids) > 10:
            print(f"  ... and {len(failed_ids)-10} more")
    else:
        print("\n+> All sources downloaded successfully!")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()