# TESScut Asteroids for ML

*A library to create ML-ready data from TESScut with asteroids*

This package has an object class and scripts to create cutouts from TESS full frame images and query JPL Horizon to find observed asteroids in the FFI images.
The resulting data contaiins the flux cube, asteroid mask in the same shape as the flux cube, and other related vectors (time, CBV, quaternions, etc) that are formated to be used by ML detection models.

# Installation

Clone this repo and then install it with pip

```
clone https://github.com/jorgemarpa/tess-asteroid-ml.git

cd tess-asteroid-ml
pip install .
```

# Example use

To create the ML ready training data use the *make_TESScut_asteroids.py* script with the desired arguments (see below). This will do the following:

1. First access the asked TESS Sector/Camera/CCD image from MAST/AWS. 
2. Given a grid pattern (sparse or dense) it uses AstroCut to make cutouts from the FFI and save them to disk. Be carefull to run the script with the argument `--download` onbly the first time for a Sector/Camera/CCD because it will query Astrocut to make and download the files. Fowllowing runs must be withouth the flag to use local data.
3. Uses the asteroid catalog to get the first guess of available asteroids in the FFI.
4. Iterates over the available cutouts and available asteroids to make the mask arrays and access other vectors (CBV, quaternions, time, etc). 
5. It saves the training data as *npz* files.

 

```
usage: make_TESScut_asteroids.py [-h] [--sector SECTOR] [--camera CAMERA] [--ccd CCD] [--cutout-size CUTOUT_SIZE] [--lim-mag LIM_MAG] [--sampling SAMPLING] [--fit-bkg] [--plot]
                                 [--verbose] [--download]

Creates a dataset from TESS FFIs. Makes 64x64 pixel cuts, uses JPL to create a asteroid mask.

optional arguments:
  -h, --help            show this help message and exit
  --sector SECTOR       TESS sector number.
  --camera CAMERA       TESS camera number.
  --ccd CCD             TESS CCD number
  --cutout-size CUTOUT_SIZE
                        Cutout size in pixels
  --lim-mag LIM_MAG     Limiting magnitude in V band.
  --sampling SAMPLING   Select a `dense` grid that covers corner to corner of the FFI or a `sparse` that uses only 7 rows from the grid.
  --fit-bkg             Fit and substract background (flag).
  --plot                Plot FFI (flag).
  --verbose             Verbose (flag).
  --download            Donwload cutouts from from AWS wh Astrocut (flag).
```