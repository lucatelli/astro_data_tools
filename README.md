# astro_data_tools
Tools for retrieving and working with astronomical image data. Retrieve FITS cutouts and PSFs from survey archives.

## query_fits_cutouts
### Hyper Suprime-Cam (HSC)
To retrieve data cutouts in format of FITS files from Subaru, we can use  the following script:  
[src/astro_data_tools/query_fits_cutouts/get_hsc_data.py](src/astro_data_tools/query_fits_cutouts/get_hsc_data.py).  
For that, firstly, you will need to creat an account in [https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/](https://hsc-release.mtk.nao.ac.jp/doc/index.php/data-access__pdr3/).

Once everything is done, the script can be used in many ways. 

#### Single Source download
```bash
python get_hsc_data.py --username <username> --password <password> --filter ALL --download-psf --download-variance --source-name LEDA1148416 --cutout-size 768 ./output/LEDA1148416
```
If one wants to obtain the data from a source using coordinates instead of the source name, that can be achievied by passing the arguments, for example, `--source-name LEDA1148416 --ra 359.112928  --dec -0.237628` instead, in which `--source-name` will only be used to name the downloaded files.

#### Catalogue Download
Multiple sources (from a catalogue) can be downloaded at once. 
```bash
python get_hsc_data.py <catalogue_file.csv> output_dir/subaru/g_band_dr3/ --filter G --username <username> --password <password> --download-psf --download-variance --cutout-size 512
```
The `catalogue_file.csv` should at least contain the following collums: `ID,RA,DEC`. 

Note: all cutout sizes with `--cutout-size` are in pixels.

### Legacy Surveys - DECam Instrument
Similarly, we can also retrieve cutouts from DECam -- Legacy Surveys [https://www.legacysurvey.org/](https://www.legacysurvey.org/), using their public service (no account needed), with the script:  
[src/astro_data_tools/query_fits_cutouts/get_legacy_survey_data.py](src/astro_data_tools/query_fits_cutouts/get_legacy_survey_data.py)

For a single source:
```bash
python get_legacy_survey_data.py --source-name LEDA1148416 --bands ALL --min-size 128 --psf --invvar --n-jobs 2 --size 512 --dr 10 output/multi_band/ls_dr10/LEDA1148416/
```
If there are failed downloads (by server response faliures), we can run again with the `--retry` flag, and only those failed cases will be attempted. 
```bash
python get_legacy_survey_data.py --source-name LEDA1148416 --bands ALL --min-size 128 --psf --invvar --n-jobs 2 --size 512 --dr 10 output/multi_band/ls_dr10/LEDA1148416/ --retry
```

To download a list of sources from a catalogue, the principle is the same as the HSC script. 

Important: Please, be aware that the service can become unavailable if multiple jobs are requested, so it is recommended that we set a number of jobs to a fair number (e.g. 2 or 3).
