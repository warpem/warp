# Processing Tilt Series Stacks

`WarpTools` expects multi-frame micrographs (movies) as input when processing
tilt series data. Sometimes only tilt series stacks are available.
This page will show you how to prepare tilt series stacks for processing in
`WarpTools`.

We will use a SARS-CoV-2 tilt series from [EMPIAR-10453](https://www.ebi.ac.uk/empiar/EMPIAR-10453/) to demonstrate.
The script may need to be modified for other datasets.

## Splitting Tilt Series Stacks

We will use a simple script to split our tilt series into individual tilt images.

### Preparation
This Python scripts depends on the Python packages
[mrcfile](https://github.com/ccpem/mrcfile),
[tifffile](https://github.com/cgohlke/tifffile)
and
[mdocfile](https://github.com/teamtomo/mdocfile).
These can be installed from [PyPI](https://pypi.org/) in your warp conda environment.

```sh
pip install mrcfile tifffile mdocfile
```

### Running the script

Below is the starting directory structure and the contents of the Python script.

```txt title="directory structure (single tilt series)"
.
├── mdoc
│   └── TS_005.mrc.mdoc
├── tiltseries
│   └── TS_005.mrc
└── warp_ts_split_10453.py
```

```python title="warp_ts_split_10453.py"
from pathlib import Path
import mrcfile
import mdocfile

# modify these variables for your own processing
OUTPUT_DIRECTORY = 'movies'
TILT_SERIES_DIRECTORY = 'tiltseries'
MDOC_DIRECTORY = 'mdoc'


# define a function for processing a single tilt series
def process_tilt_series(mdoc_file: Path, output_directory: Path):
    # load mdoc into pandas dataframe
    df = mdocfile.read(mdoc_file)

    # load tilt series into numpy array
    tilt_series_file = Path(TILT_SERIES_DIRECTORY) / mdoc_file.stem
    tilt_series = mrcfile.read(tilt_series_file)

    # check for same number of tilts in mdoc and image file
    if len(df) != len(tilt_series):
        e = f'more tilts in mdoc than tilt series for {mdoc_file}'
        raise RuntimeError(e)

    # split tilt series into individual tilts
    for movie_path, image in zip(df['SubFramePath'], tilt_series):
        output_movie_file = output_directory / Path(movie_path.name)
        if output_movie_file.suffix.lower() == '.mrc':
            mrcfile.write(output_movie_file, image)
        elif output_movie_file.suffix in ('.tif', '.tiff'):
            tifffile.imwrite(output_movie_file, image)

# main program which loops over all tilt series...
if __name__ == '__main__':
    # create output directory
    Path(OUTPUT_DIRECTORY).mkdir(exist_ok=True, parents=True)
    
    # get list of mdoc files to process
    mdocs = list(Path(MDOC_DIRECTORY).glob('*.mdoc'))
    
    # process mdoc files one by one
    for mdoc in mdocs:
        process_tilt_series(mdoc_file=mdoc, output_directory=Path(OUTPUT_DIRECTORY))
```

```sh
python warp_ts_split_10453.py
```

## Output

```txt title="output directory structure"
.
├── mdoc
│   └── TS_005.mrc.mdoc
├── movies
│   ├── TS_005_001_-0.0.tif
│   ├── TS_005_002_3.0.tif
│   ├── TS_005_003_6.0.tif
│   ├── TS_005_004_-3.0.tif
│   ├── TS_005_005_-6.0.tif
│   ├── TS_005_006_9.0.tif
│   ├── TS_005_007_12.0.tif
│   ├── TS_005_008_-9.0.tif
│   ├── TS_005_009_-12.0.tif
│   ├── TS_005_010_15.0.tif
│   ├── TS_005_011_18.0.tif
│   ├── TS_005_012_-15.0.tif
│   ├── TS_005_013_-18.0.tif
│   ├── TS_005_014_21.0.tif
│   ├── TS_005_015_24.0.tif
│   ├── TS_005_016_-21.0.tif
│   ├── TS_005_017_-24.0.tif
│   ├── TS_005_018_27.0.tif
│   ├── TS_005_019_30.0.tif
│   ├── TS_005_020_-27.0.tif
│   ├── TS_005_021_-30.0.tif
│   ├── TS_005_022_33.0.tif
│   ├── TS_005_023_36.0.tif
│   ├── TS_005_024_-33.0.tif
│   ├── TS_005_025_-36.0.tif
│   ├── TS_005_026_39.0.tif
│   ├── TS_005_027_42.0.tif
│   ├── TS_005_028_-39.0.tif
│   ├── TS_005_029_-42.0.tif
│   ├── TS_005_030_45.0.tif
│   ├── TS_005_031_48.0.tif
│   ├── TS_005_032_-45.0.tif
│   ├── TS_005_033_-48.0.tif
│   ├── TS_005_034_51.0.tif
│   ├── TS_005_035_54.0.tif
│   ├── TS_005_036_-51.0.tif
│   ├── TS_005_037_-54.0.tif
│   ├── TS_005_038_57.0.tif
│   ├── TS_005_039_60.0.tif
│   ├── TS_005_040_-57.0.tif
│   └── TS_005_041_-60.0.tif
├── tiltseries
│   └── TS_005.mrc
└── warp_ts_split_10453.py
```

## Subsequent Processing

You should now be able to follow the
[quick start guide for tilt series processing](../../user_guide/warptools/quick_start_warptools_tilt_series.md).
Remember to process your data as single frame images!