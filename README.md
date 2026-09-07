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