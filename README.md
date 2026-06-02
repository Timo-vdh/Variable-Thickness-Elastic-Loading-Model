# Variable Thickness Elastic Loading Model
Development of a variable thickness, thin spherical shell elastic loading model for the lithosphere of Mars. Part of MSc Thesis work at DLR Berlin for the TU Delft

Three primary models currently worked on with three different approaches for solving the variable thickness equations with no tangential loading:

M1: **M1VarD_SPAT.py**
- Solves the operator A(D;w) and A(alpha;F) by converting it into spatial domain with a first guess of w and F based on the constant thickness model.
- Takes the spatial derivatives of D(theta,phi), w(theta,phi), alpha(theta,phi) and F(theta,phi).

M2: **M1VarD_LP.py**
- Similar as 1, but now the unknown coefficients w and F are rewritten into the spherical harmonic notation: w(theta,phi) = Sum(lm) w_lm Y_lm.
- This way only the spatial derivatives of the Spherical harmonic functions (and therefore the Legendre polynomials, LP) are required, as they are the same for all parameters in A.

M3: **M1VarD_SPEC_opt.py** / **TSA_spectral_corrected.py**
- Solves the set of equations for w_lm and F_lm directly in the spectral domain following the approach of Kalousova et al. (2012).


All three models are still under development, with M3 having the first priority currently.
