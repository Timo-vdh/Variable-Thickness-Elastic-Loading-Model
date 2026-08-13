# Variable Thickness Elastic Loading Model
Development of a variable thickness, thin spherical shell elastic loading model for the lithosphere of Mars. The thickness of the shell is the elastic lithosphere thickness Te. This model is developed as part of Aerospace Engineering MSc Thesis work at DLR Berlin for the TU Delft. Flexure model is built using the theory of Beuthe (2008) and Beuthe (2010) and with the spectral notation of Kalousova et al. (2012). Equations for tangential loading and internal variations (bottom loads) are taken largely from the Displacement_strain_planet (DSP) model from Broquet (2024) and the paper of Broquet & Andrews-Hanna (2023).

The models are developed iteratively by adding more physics after each model is verified against the DSP model in a constant thickness case. The currently most complete model, M4, implements top and bottom loads as well as tangential loading (consoidal potential only, no toroidal potential). It also aims to calculate the stresses and strains at the surface of Mars, but this is still a work in progress.

To run the model it is required to compute Gaunt coefficients, which are then stored to a local drive. These files can become large depending on the lmax that is solved for, reaching 59 GB at lmax=90.

