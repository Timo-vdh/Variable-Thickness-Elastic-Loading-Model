# Variable Thickness Elastic Loading Model
A thin spherical shell elastic loading model for planetary elastic lithospheres, capable of incorporating lateral variations in the elastic thickness Te. 

## Description
The VarTeModel is a Python code that can be used to compute planetary gravity, flexural displacement, crustal thickness variations, internal density variations, stresses and strains in spherical harmonics. The model can be used to invert observed gravity and topography for internal variations in crustal thickness or density, or alternatively, by constraining both these internal variations, used to forward model a gravity field.

The flexure model is built based upon the theoretical framework of [Banerdt (1986)][banerdt1986] and [Beuthe (2008)][beuthe2008], with the spectral formulations of [Beuthe (2010)][beuthe2010] and [Kalousová et al. (2012)][kalousova2012] and corrections of of [Broquet & Andrews-Hanna (2022)][broquet2022] and the [Displacement_strain_planet][dsp] (DSP) model of [Broquet (2024)][broquet2024].

To run the model, it is required to once compute Gaunt coefficients, which are then stored to a local drive. These files can become large depending on the maximum spherical harmonic degree that is solved for, reaching 38 GB at lmax=90.

## Limitations
- The model is discontinuous at points where the elastic thickness transitions from deeper to shallower than the mean crustal thickness, which causes a small inconsistency in the model outputs. Future efforts on improving this can be done by creating a smooth transition at these regions.
- The model does not incorporate toroidal loading (V in [Beuthe, 2008][beuthe2008]) as a relation for this was not known during development
- The model does not incorporate a formulation for the differential operator **B** of [Beuthe (2008)][beuthe2008], as this too was not known during the development of the model, causing also toroidal displacements to be absent in the formulation. Future improvements should focus on deriving a spectral notation and implementing this operator completely. These additions are simple additions to the already present terms.

## Dependencies
The model heavily relies on the [pyshtools][pyshtools] package of ([Wieczorek & Meschede, 2018][wieczorek2018]), used extensively for the spherical harmonic transforms, Gaunt coefficient computations and gravitational potential finite amplitude calculations. Additionally, the model is developed with much inspiration by the [Displacement_strain_planet][dsp] package of [Broquet (2024)][broquet2024] and also uses several functions of this code.

The variable thickness formulation may be incorporated into the DSP package in future developments and improvements, allowing for a full and centralized location for planetary gravity, crustal thickness, displacement, stress and strain calculations in spherical harmonics.

Versions of dependency codes for which the model is verified:
- Python 3.10
- pyshtools: 4.14.1
- DSP: 0.6.3 (use develop branch)

## Benchmarks
To the author's knowledge there is no other planetary elastic loading model that incorporates lateral variations in the elastic thickness, preventing full verification of 2D Te variability. Nevertheless, several benchmarks are performed:
- The constant Te limit against [Displacement_strain_planet][dsp] [(Broquet, 2024)][broquet2024].
- Te varying with latitude against the formulation of [Kalousová et al. (2012)][kalousova2012]. This covers 1D Te variations only.
- Constant Te stress fields against [Andrews-Hanna et al. (2008)][andrewshanna2008].

## Files
`VarTeModel.py` contains all the elastic loading functionalities and computations for gravity, displacements, internal variations, stresses and strains. The model can be solved by calling only `solve(topo_clm, geoid_clm, Te_clm)`, after which the loading equations are solved consistently depending on which parameter is solved for (either crustal thickness variations, internal density variations, or geoid height).

`Gaunt_utils_v5.py` contains the framework for constructing, saving, and loading the Gaunt coefficients in an optimized way using structure of arrays (SoA).

`Te_twoval_builder.py` contains several simple functions to create a synthetic two-step Te map that can be used for simplified variable Te assessments.

## How to install and run VarTeModel
Clone the repository with git:

    git clone https://github.com/Timo-vdh/Variable-Thickness-Elastic-Loading-Model.git
    cd Variable-Thickness-Elastic-Loading-Model

or download it as a zip file and unzip.
The required packages can be installed with pip:

    pip install -r requirements.txt

## Examples
One example file (VarTeModel_example_runner.py) is created to show how to initialize the model when inverting the observed gravity and topography for lateral crustal thickness variations, assuming no lateral density variations are present.

The example can be run with

    python VarTeModel_example_runner.py
    
or open `VarTeModel_example_runner.py` in a python GUI and run it from the repository folder.
On the first run, the Gaunt coefficients are computed and cached locally; this can take a while for high `lmax`. Later runs load the cached file.

## Author
Timo van den Heuvel (Timo-vdh, timo.vandenheuvel@dlr.de)

## Acknowledgements
This code has been developed for my MSc Thesis work at Delft University of Technology and DLR Berlin. I would like to greatly thank Adrien Broquet and Bart Root for their extensive supervision and advice throughout the development and improvement of the code.



[banerdt1986]: https://doi.org/10.1029/JB091iB01p00403
[beuthe2008]: https://doi.org/10.1111/j.1365-246X.2007.03671.x
[beuthe2010]: https://doi.org/10.1016/j.icarus.2010.04.019
[kalousova2012]: https://doi.org/10.1111/j.1365-246X.2012.05539.x
[broquet2022]: https://doi.org/10.1038/s41550-022-01836-3
[broquet2024]: https://doi.org/10.5281/zenodo.4916799
[andrewshanna2008]: https://doi.org/10.1029/2007JE002980
[wieczorek2018]: https://doi.org/10.1029/2018GC007529
[pyshtools]: https://github.com/SHTOOLS/SHTOOLS
[dsp]: https://github.com/AB-Ares/Displacement_strain_planet
