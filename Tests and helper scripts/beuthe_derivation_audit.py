# =============================================================================
#  Beuthe (2008) variable-thickness flexure — symbolic derivation audit
#  Equations 2.27 - 2.97  (M2 tangential loading + M3 dc/drho extension)
#
#  This script re-derives every step from the spectral Kalousova system through
#  the final M2 and M3 operators, and PRINTS the canonical (sign-corrected)
#  final equations.  A dedicated verification block at the end checks a
#  user-supplied A-tilde / b-tilde against the derived-correct forms, term by
#  term, so it can be reused as a regression check and as documentation.
#
#  Convention notes
#  ----------------
#  * LAP  = output-degree Laplacian      -l(l+1)        (Beuthe Delta)
#  * LAP2 = output-degree Laplacian + 2  -l(l+1)+2      (Beuthe Delta')
#  * v1v  = nu/(1-nu)
#  * phi  = (R - Tc)/R                    (Broquet geoid radius ratio)
#  * maxT = max((Te)_LM - Tc, 0)          (treated as an independent field)
#  * The 1/(E Te0^3) scaler (eq-1) and the R scaler (eq-2) are factored where
#    indicated; sign analysis is independent of these positive scalers.
#  * For sign verification, LAP/LAP2/Te/Tc/maxT/H/w/G/alpha are scalar symbols.
#    The field STRUCTURE (which Te sits at LM vs lm, where the Laplacian acts)
#    is carried as an annotation string and does not affect the coefficient
#    signs that this audit targets.
# =============================================================================

import sympy as sp

# ----------------------------------------------------------------------------
# Symbols
# ----------------------------------------------------------------------------
nu, E              = sp.symbols('nu E', positive=True)
rhoc, rhom, rhobar = sp.symbols('rho_c rho_m rhobar', positive=True)
g0, gm, gM         = sp.symbols('g0 g_m g_M', positive=True)
R, Tc, Te, Te0, M  = sp.symbols('R T_c T_e T_e0 M', positive=True)
H, w, G, alpha     = sp.symbols('H w G alpha', real=True)
maxT               = sp.symbols('maxT', real=True)          # max((Te)_LM - Tc, 0)
maxTMt, minM_TMt   = sp.symbols('maxTMt minM_TMt', real=True)
l                  = sp.symbols('l', positive=True, integer=True)
LAP                = sp.symbols('LAP', real=True)           # -l(l+1)
LAP2               = sp.symbols('LAP2', real=True)          # -l(l+1)+2
Mt, Mb             = sp.symbols('Mt Mb', real=True)

v1v  = nu/(1 - nu)
drho_cm = rhom - rhoc
phi  = (R - Tc)/R
xi   = 12*R**2/Te**2                                        # eq 2.46

def banner(txt):
    print("\n" + "=" * 78 + f"\n{txt}\n" + "=" * 78)

def show(name, expr):
    print(f"  {name} = {sp.factor(sp.expand(expr))}")

# =============================================================================
#  PART A — M2 :  tangential loading with  dc = 0,  drho = 0   (eqs 2.44-2.62)
# =============================================================================
banner("PART A — M2  (eqs 2.44-2.62):  tangential loading, dc = drho = 0")

# A.0 -- Starting spectral system (Kalousova et al. 2012), eqs 2.27-2.43
#        sum_{l'm'} A^{lm}_{l'm'} w + a_l F = y_lm
#        sum_{l'm'} B^{lm}_{l'm'} F + b_l w = 0
#        These are taken as given (verified separately against Beuthe 2010).
#        a_l = (R/Te0)^3 (LAP2)/E      (2.38/2.90),   b_l = -(LAP2) (2.42/2.92)
print("\n[A.0] Base Kalousova system (2.27-2.43) assumed correct (Beuthe 2010 A-operator).")
print("      a_l = (R/Te0)^3 (LAP2)/E ;  b_l = -(LAP2) = l(l+1)-2.")

# A.1 -- Beuthe Omega-extended flexure equations (2.44-2.45):
#        Delta'(D Delta' w) - (1-nu)A(D;w) + R^3 Delta' F = -R^4 q - 2R^3 Omega + (R^3/xi) Delta Omega
#        Delta'(a Delta' F) - (1+nu)A(a;F) - (1/R) Delta' w = -(1-nu) Delta'(a Omega)
#        Banerdt/Broquet consoidal load (2.47), reduced to dc=drho=0 (2.48):
ROmega_H_M2 = v1v*rhoc*g0*Te                                    # coeff of H in R*Omega
Bw          = rhoc*gm*(v1v*Te - Tc) - rhom*gm*maxT             # w-bracket of Omega (2.48)
ROmega_w_M2 = -Bw                                               # coeff of w in R*Omega (note minus!)
print("\n[A.1] R*Omega (Banerdt->Beuthe, *R removes 1/R), eqs 2.47-2.49:")
show("R*Omega : H-coeff", ROmega_H_M2)
show("R*Omega : w-coeff", ROmega_w_M2)

# A.2 -- Substitute R*Omega into the master system (2.50-2.51).  Row-1 prefactor:
#        pref = -2R^3 + (R^3/xi) LAP   and   R^3/xi = R Te^2/12   (verify):
pref = -2*R**3 + (R**3/xi)*LAP
assert sp.simplify(pref - (-2*R**3 + R*Te**2/12*LAP)) == 0
pref = -2*R**3 + R*Te**2/12*LAP
print("\n[A.2] Row-1 Omega prefactor pref = -2R^3 + (R^3/xi)LAP = -2R^3 + (R Te^2/12)LAP  [checked].")

# A.3 -- Operator construction rule (the crux of the sign bookkeeping)
#        Row 1:  sum A w + a F = y + pref * R*Omega
#                H,G parts stay on RHS  -> c_lm   (single sign)
#                w part moves to LHS    -> A-tilde = -(pref * w-coeff)   (one extra minus)
#        Row 2:  sum B F + b w = -(1-nu) LAP2 (alpha R*Omega)
#                H,G parts stay on RHS  -> d_lm
#                w part moves to LHS    -> b-tilde = +(1-nu) LAP2 alpha (w-coeff)
#        (Scalers: 1/(E Te0^3) on eq-1 ; R on eq-2.)
def make_Atilde(cW):                       # cW = w-coefficient of R*Omega
    return sp.expand(-pref*cW)             # pre 1/(E Te0^3)
def make_btilde(cW):
    return sp.expand(R*(1-nu)*LAP2*alpha*cW)   # pre nothing (R kept explicit, as in doc)
def make_clm(cH_plus_cG):
    return sp.expand(pref*cH_plus_cG)      # pre 1/(E Te0^3)
def make_dlm(cH_plus_cG):
    return sp.expand(-R*(1-nu)*LAP2*alpha*cH_plus_cG)

# A.4 -- M2 final operators (corrected).  Print canonical forms (2.58-2.61):
Atilde_M2 = make_Atilde(ROmega_w_M2)
btilde_M2 = make_btilde(ROmega_w_M2)
clm_M2    = make_clm(ROmega_H_M2*H)        # H only (G absent in M2)
dlm_M2    = make_dlm(ROmega_H_M2*H)

print("\n[A.4] CANONICAL CORRECTED M2 OPERATORS")
print("      (pre-scalers: A-tilde,c_lm carry 1/(E Te0^3); b-tilde,d_lm carry their R)")
show("A-tilde  (eq 2.58, corrected)", Atilde_M2)
show("b-tilde  (eq 2.59, corrected)", btilde_M2)
show("c_lm     (eq 2.60)           ", clm_M2)
show("d_lm     (eq 2.61)           ", dlm_M2)

# Human-readable term list for A-tilde (M2) so signs are explicit:
print("\n      A-tilde(M2) term-by-term (×1/(E Te0^3)):")
M2_Atilde_terms = [
 ("Te        (R^3 )", -2*R**3*rhoc*gm*v1v,   "(Te)_LM"),
 ("Tc const  (R^3 )", +2*R**3*rhoc*gm,       "Tc  [diagonal]"),
 ("max       (R^3 )", +2*R**3*rhom*gm,       "max"),
 ("Te2 LAP Te(R/12)", +sp.Rational(1,12)*R*rhoc*gm*v1v, "(Te)_LM^2{LAP (Te)_lm}"),
 ("Tc  LAP   (R/12)", -sp.Rational(1,12)*R*rhoc*gm,     "(Te)_LM^2 LAP"),
 ("max LAP   (R/12)", -sp.Rational(1,12)*R*rhom*gm,     "(Te)_LM^2{LAP max}"),
]
for lab, c, fld in M2_Atilde_terms:
    print(f"        {lab:18s}: {str(sp.factor(c)):28s} * {fld}")

# =============================================================================
#  PART B — M3 :  include dc, set drho = 0   (eqs 2.63-2.97)
# =============================================================================
banner("PART B — M3  (eqs 2.63-2.97):  crustal-thickness variations dc, drho = 0")

# B.1 -- Broquet geoid at surface G (2.65) and at crust base Gc (2.66), drho=0.
pc = 3/(rhobar*(2*l + 1))
G_eq  = pc*( rhoc*H + drho_cm*phi**(l+2)*w - drho_cm*phi**(l+2)*sp.Symbol('dc') )       # 2.65
print("\n[B.1] Surface geoid (2.65) and crust-base geoid (2.66) taken from Broquet&AH.")

# B.2 -- Solve G for dc (2.67-2.69):
dc_sym = sp.Symbol('dc')
dc_solved = sp.solve(sp.Eq(G, G_eq), dc_sym)[0]
dc_canon  = 1/(drho_cm*phi**(l+2))*( rhoc*H - rhobar*(2*l+1)/sp.Integer(3)*G ) + w  # 2.69
assert sp.simplify(dc_solved - dc_canon) == 0
print("[B.2] Solved dc from surface geoid (2.69)  [checked == solve()]:")
print(f"      dc = 1/(drho_cm*phi^(l+2)) [ rho_c*H - rhobar(2l+1)/3 * G ] + w")

# Crust-base geoid Gc, rewritten with the buoyancy exponent.
Gc_eq = pc*( rhoc*phi**(l+1)*H + drho_cm*phi**3*(w - dc_canon) )               # 2.71
print(f"[B.2] Gc buoyancy exponent set to phi^(3)")

# B.3 -- Substitute dc into R*Omega (2.72-2.78).  The dc term in Omega is
#        -(v1v) drho_cm gm maxT * dc ; dc carries a '+w' tail and H,G content.
dc_term_coeff = -v1v*drho_cm*gm*maxT
ROmega_M3 = ( ROmega_H_M2*H + ROmega_w_M2*w + dc_term_coeff*dc_canon )
ROmega_M3 = sp.expand(ROmega_M3)
ROmega_H_M3 = sp.simplify(ROmega_M3.coeff(H))
ROmega_w_M3 = sp.simplify(ROmega_M3.coeff(w))
ROmega_G_M3 = sp.simplify(ROmega_M3.coeff(G))
print("\n[B.3] R*Omega after dc-substitution (2.72-2.78):")
show("R*Omega(M3) : H-coeff", ROmega_H_M3)
show("R*Omega(M3) : w-coeff", ROmega_w_M3)
show("R*Omega(M3) : G-coeff", ROmega_G_M3)
print("      -> the dc '+w' tail adds the term  -(v1v) drho_cm gm maxT  to the w-coeff")
print("         (this is ERROR #3's location: it must stay NEGATIVE, see eq 2.74).")

# B.4 -- Loading q_lm chain (2.79-2.86): substitute dc and Gc, drho=0.
#        q_lm = g0 rho_c (H-G) + g_m drho_cm (w - dc - Gc)
qlm = g0*rhoc*(H - G) + gm*drho_cm*(w - dc_canon - Gc_eq)
qlm = sp.expand(qlm)
qlm_H = sp.simplify(qlm.coeff(H))
qlm_G = sp.simplify(qlm.coeff(G))
qlm_w = sp.simplify(qlm.coeff(w))
print("\n[B.4] Loading q_lm after dc,Gc substitution (2.81-2.86):")
print(f"      w-coeff = {qlm_w}   (must be 0: buoyant-w cancels, removing original A buoyancy)")
assert qlm_w == 0
show("q_lm : H-coeff", qlm_H)
show("q_lm : G-coeff", qlm_G)

# B.5 -- Final M3 operators (corrected).  Build via the SAME rule as M2.
Atilde_M3 = make_Atilde(ROmega_w_M3)
btilde_M3 = make_btilde(ROmega_w_M3)
clm_M3    = make_clm(ROmega_H_M3*H + ROmega_G_M3*G)
dlm_M3    = make_dlm(ROmega_H_M3*H + ROmega_G_M3*G)

print("\n[B.5] CANONICAL CORRECTED M3 OPERATORS")
show("A-tilde (eq 2.93, corrected)", Atilde_M3)
show("b-tilde (eq 2.94, corrected)", btilde_M3)
show("c_lm    (eq 2.96)           ", clm_M3)
show("d_lm    (eq 2.97)           ", dlm_M3)
print("\n      q_lm (eq 2.95, -R^4 q is the RHS load) carries the H,G coeffs above,")
print("      with the (w-dc) buoyancy sign FIXED (ERROR #4): H-term +phi^(1-l), G-term -phi^(1-l).")

# Human-readable canonical M3 A-tilde term list (the reference for verification):
print("\n      A-tilde(M3) term-by-term (×1/(E Te0^3)):")
M3_Atilde_terms = [
 ("Te         (R^3 )", -2*R**3*rhoc*gm*v1v,            "(Te)_LM"),
 ("max        (R^3 )", +2*R**3*rhom*gm,                "max"),
 ("drho_cm*max   (R^3 )", -2*R**3*v1v*drho_cm*gm,            "max   <-- key term3"),
 ("Te2 LAP Te (R/12)", +sp.Rational(1,12)*R*rhoc*gm*v1v, "(Te)_LM^2{LAP (Te)_lm}"),
 ("Tc  LAP    (R/12)", -sp.Rational(1,12)*R*rhoc*gm,     "(Te)_LM^2 LAP"),
 ("max LAP    (R/12)", -sp.Rational(1,12)*R*rhom*gm,     "(Te)_LM^2{LAP max}"),
 ("drho_cm*maxLAP(R/12)", +sp.Rational(1,12)*R*v1v*drho_cm*gm, "(Te)_LM^2{LAP max}"),
 ("Tc const   (R^3 )", +2*R**3*rhoc*gm,                "Tc  [diagonal]"),
]
for lab, c, fld in M3_Atilde_terms:
    print(f"        {lab:18s}: {str(sp.factor(c)):30s} * {fld}")

print("\n      b-tilde(M3) term-by-term (×R):")
M3_btilde_terms = [
 ("Te      ", -nu*rhoc*gm,        "(Te)_lm alpha_LM  [ (1-nu)v1v = nu absorbed ]"),
 ("Tc      ", (1-nu)*rhoc*gm,     "alpha_LM * Tc"),
 ("max     ", (1-nu)*rhom*gm,     "max * alpha_LM"),
 ("drho_cm*max", -nu*drho_cm*gm,        "max * alpha_lm   <-- key term4"),
]
for lab, c, fld in M3_btilde_terms:
    print(f"        {lab:10s}: {str(sp.factor(c)):22s} * (LAP2) {fld}")

# =============================================================================
#  PART C — VERIFY THE USER-SUPPLIED M3  A-tilde  AND  b-tilde
# =============================================================================
banner("PART C — verification of the supplied (corrected) M3 A-tilde and b-tilde")

# Helper: compare a supplied {label: coeff} against the canonical reference.
def verify(title, supplied, reference):
    print(f"\n  {title}")
    allok = True
    for lab, ref_c in reference:
        sup_c = supplied[lab]
        ok    = sp.simplify(sup_c - ref_c) == 0
        flip  = sp.simplify(sup_c + ref_c) == 0
        tag   = "OK" if ok else ("SIGN-FLIP" if flip else "MISMATCH")
        if not ok: allok = False
        print(f"    [{tag:9s}] {lab:18s} supplied={str(sp.factor(sup_c)):30s} expected={str(sp.factor(ref_c))}")
    print(f"    => {'ALL TERMS CORRECT' if allok else 'NEEDS CORRECTION (see flags)'}")
    return allok

# ---- Supplied A-tilde (M3) as pasted by the user (pre 1/(E Te0^3)) ----
USER_Atilde = {
 "Te         (R^3 )": -2*R**3*rhoc*gm*v1v,
 "max        (R^3 )": +2*R**3*rhom*gm,
 "drho_cm*max   (R^3 )": -2*R**3*v1v*drho_cm*gm,
 "Te2 LAP Te (R/12)": +sp.Rational(1,12)*R*rhoc*gm*v1v,
 "Tc  LAP    (R/12)": -sp.Rational(1,12)*R*rhoc*gm,
 "max LAP    (R/12)": -sp.Rational(1,12)*R*rhom*gm,
 "drho_cm*maxLAP(R/12)": +sp.Rational(1,12)*R*v1v*drho_cm*gm,
 "Tc const   (R^3 )": +2*R**3*rhoc*gm,
}
REF_Atilde = [(lab, c) for (lab, c, _f) in M3_Atilde_terms]
verify("A-tilde (M3):", USER_Atilde, REF_Atilde)

# ---- Supplied b-tilde (M3) as pasted by the user (pre R) ----
USER_btilde = {
 "Te      ": -nu*rhoc*gm,
 "Tc      ": (1-nu)*rhoc*gm,
 "max     ": (1-nu)*rhom*gm,
 "drho_cm*max": -nu*drho_cm*gm,                            # user flipped this to -nu
}
REF_btilde = [(lab, c) for (lab, c, _f) in M3_btilde_terms]
verify("b-tilde (M3):", USER_btilde, REF_btilde)

banner("DONE")







# %%





# =============================================================================
#  PART D — M4 :  include drho_lm, set dc = 0   (newly derived equations)
# =============================================================================
banner("PART D — M4  (new equations):  mantle density variations drho_lm, dc = 0")


# B.1 -- Broquet geoid at surface G (2.65) and at crust base Gc (2.66), dc=0.
pc = 3/(rhobar*(2*l + 1))
G_eq  = sp.Symbol('pc')*( rhoc*H + drho_cm*phi**(l+2)*w 
            # - drho_cm*phi**(l+2)*sp.Symbol('dc') 
            + R/(l+3) * ( ((R-Mt)/R)**(l+3) - ((R-Mb)/R)**(l+3) ) *sp.Symbol('drho_lm'))       # 2.65
print("\n[B.1] Surface geoid (2.65) and crust-base geoid (2.66) taken from Broquet&AH.")

Ks, Kb = sp.symbols('K_surf K_base')
# G_eq  = pc*(rhoc*H 
#             + drho_cm*phi**(l+2)*w 
#             + Ks*sp.Symbol('drho_lm'))

# B.2 -- Solve G for drho_lm:
drho_lm_sym = sp.Symbol('drho_lm')
drho_lm_sym_solved = sp.solve(sp.Eq(G, G_eq), drho_lm_sym)[0]
print("[B.2] Solved drho_lm from surface geoid")
print("G_lm solved for drho_lm in latex format: ", sp.latex((drho_lm_sym_solved)))

# Crust-base geoid Gc, dc=0
Gc_eq = pc*( rhoc*phi**(l+1)*H 
            + drho_cm*phi**3*w 
            # - drho_cm*phi**3*sp.Symbol('dc'))
            + R/(l+3) *  (( ((R-Tc)/(R-Mt))**(l+1) - (R-Mt)**3 )/((R-Tc)*R**2) 
                          - ( ((R-Tc)/(R-Mb))**(l+1) - (R-Mb)**3 )/((R-Tc)*R**2) ) 
                            * sp.Symbol('drho_lm')  )                # 2.71
Gc_eq = pc*(rhoc*phi**(l+1)*H + drho_cm*phi**3*w + Kb*sp.Symbol('drho_lm'))



# B.3 -- Substitute drho_lm into R*Omega.  
#        The dc term in Omega is -(v1v) drho_cm gm maxT * dc ; 
#        dc carries a '+w' tail and H,G content.
#        The drho_lm term in Omega is -1/2 (v1v) gm maxTMt minM_TMt * drho_lm
#        drho_lm carries a '+w' tail and H,G content.
drho_lm_term_coeff = -sp.Rational(1, 2)*(v1v)*gm*maxTMt*minM_TMt
ROmega_M4 = ( ROmega_H_M2*H + ROmega_w_M2*w + drho_lm_term_coeff*drho_lm_sym_solved )
ROmega_M4 = sp.expand(ROmega_M4)
ROmega_H_M4 = sp.simplify(ROmega_M4.coeff(H))
ROmega_w_M4 = sp.simplify(ROmega_M4.coeff(w))
ROmega_G_M4 = sp.simplify(ROmega_M4.coeff(G))
print("\n[B.3] R*Omega after drho_lm-substitution:")
show("R*Omega(M4) : H-coeff", ROmega_H_M4)
show("R*Omega(M4) : w-coeff", ROmega_w_M4)
show("R*Omega(M4) : G-coeff", ROmega_G_M4)
print("      -> the dc '+w' tail adds the term  -1/2 (v1v) gm maxTMt minM_TMt  to the w-coeff")




# B.4 -- Loading q_lm chain: substitute drho_lm and Gc, dc=0.
#        q_lm = g0 rho_c (H-G) + g_m drho_cm (w - dc - Gc)
#        q_lm = g0 rho_c (H-G) + g_m drho_cm (w - Gc) + g_M M drho_lm
qlm_M4 = g0*rhoc*(H - G) + gm*drho_cm*(w - Gc_eq) + gM*M*drho_lm_sym_solved
qlm_M4 = sp.expand(qlm_M4)
qlm_H_M4 = sp.simplify(qlm_M4.coeff(H))
qlm_G_M4 = sp.simplify(qlm_M4.coeff(G))
qlm_w_M4 = sp.simplify(qlm_M4.coeff(w))
print("\n[B.4] Loading q_lm after Gc,drho_lm substitution:")
print(f"      w-coeff = {qlm_w_M4}")
# assert qlm_w == 0
show("q_lm_M4 : H-coeff", qlm_H_M4)
show("q_lm_M4 : G-coeff", qlm_G_M4)
show("q_lm_M4 : w-coeff", qlm_w_M4)



# B.5 -- Final M3 operators (corrected).  Build via the SAME rule as M2.
Atilde_M4 = make_Atilde(ROmega_w_M4)
btilde_M4 = make_btilde(ROmega_w_M4)
clm_M4    = make_clm(ROmega_H_M4*H + ROmega_G_M4*G)
dlm_M4    = make_dlm(ROmega_H_M4*H + ROmega_G_M4*G)

Atilde_M4 = sp.factor(sp.nsimplify(Atilde_M4))
btilde_M4 = sp.factor(sp.nsimplify(btilde_M4))
clm_M4    = sp.factor(sp.nsimplify(clm_M4))
dlm_M4    = sp.factor(sp.nsimplify(dlm_M4))


print("\n[B.5] M4 OPERATORS")
show("A-tilde   ", (Atilde_M4))
show("\nb-tilde ", (btilde_M4))
show("\nc_lm    ", (clm_M4))
show("\nd_lm    ", (dlm_M4))
print("\n      q_lm (-R^4 q is the RHS load) carries the H,G coeffs above,")
# print("      with the (w-dc) buoyancy sign FIXED (ERROR #4): H-term +phi^(1-l), G-term -phi^(1-l).")



banner("LATEX EXPORT")
# We use long_frac_ratio=2 so SymPy uses proper vertical fraction bars,
# and mul_symbol='dot' so it doesn't run variable names together.
latex_Atilde_M4 = sp.latex(Atilde_M4, long_frac_ratio=2)
latex_btilde_M4 = sp.latex(btilde_M4, long_frac_ratio=2)
latex_clm_M4    = sp.latex(clm_M4, long_frac_ratio=2)
latex_dlm_M4    = sp.latex(dlm_M4, long_frac_ratio=2)

print("A-tilde in LaTeX format:\n", latex_Atilde_M4, "\n")
print("b-tilde in LaTeX format:\n", latex_btilde_M4, "\n")
print("clm in LaTeX format:\n", latex_clm_M4, "\n")
print("dlm in LaTeX format:\n", latex_dlm_M4, "\n")




