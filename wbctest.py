import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------
# --------------- 1D wave equation stability analysis ------------------------------------------------
# ----------------------------------------------------------------------------------------------------
print('1D wave equation stability analysis')
# pde
u_tt, u_xx, c = sp.symbols( 'u_tt,u_xx,c' )
pde = sp.Eq(u_tt, c**2 * u_xx)
# discretization
# ujn, j ~ position, n ~ time
u00, u01, u02, u10, u11, u12, u20, u21, u22 = sp.symbols( 'u00,u01,u02,u10,u11,u12,u20,u21,u22' )
dx, dt = sp.symbols( 'dx,dt' )
pdeD = pde.subs( u_tt, ( u12 - 2*u11 + u10 ) / dt**2 )
pdeD = pdeD.subs( u_xx, ( u21 - 2*u11 + u01 ) / dx**2 )
# substitute one mode of the error
e0, e1, e2, k = sp.symbols( 'e0,e1,e2,k' )
uD_to_error = [(u00,e0*sp.exp(-sp.I*k*dx)), (u01,e1*sp.exp(-sp.I*k*dx)), (u02,e2*sp.exp(-sp.I*k*dx)), (u10,e0), (u11,e1), (u12,e2), (u20,e0*sp.exp(sp.I*k*dx)), (u21,e1*sp.exp(sp.I*k*dx)), (u22,e2*sp.exp(sp.I*k*dx))]
pdeDE = pdeD.subs( uD_to_error )
# char. eq. of difference eq. with char. roots r1 r2
r = sp.symbols('r')
theta = sp.symbols('theta', real=True)

expr = pdeDE.subs([
    (e2, r**2),
    (e1, r),
    (e0, 1)
])

expr = expr.subs(sp.exp(sp.I*k*dx), sp.exp(sp.I*theta))
expr = expr.subs(sp.exp(-sp.I*k*dx), sp.exp(-sp.I*theta))

expr = sp.simplify(expr)

# convert to cosine
expr = expr.replace(
    lambda x: x.has(sp.exp(sp.I*theta)) or x.has(sp.exp(-sp.I*theta)),
    lambda x: sp.simplify(x.rewrite(sp.cos))
)

expr = sp.simplify(expr.rewrite(sp.cos))

lam = sp.symbols('lam', real=True)

expr = expr.subs(c**2 * dt**2 / dx**2, lam**2)

expr = sp.simplify(sp.factor(expr))

poly = sp.Poly(expr, r)

a2 = poly.coeffs()[0]   # coefficient of r^2
a1 = poly.coeffs()[1]   # coefficient of r
a0 = poly.coeffs()[2]   # constant

a2, a1, a0 = poly.all_coeffs()

expr_norm = expr / a2
poly = sp.Poly(expr_norm, r)

_, a1, a0 = poly.all_coeffs()

a = -a1 / 2

a = sp.simplify(a)

theta = sp.symbols('theta', real=True)

a_expr = 1 - lam**2 * (1 - sp.cos(theta))

# worst case: cos(theta) = -1
a_worst = a_expr.subs(sp.cos(theta), -1)

sp.solve_univariate_inequality(sp.Abs(a_worst) <= 1, lam)

# 0 <= lam**2 <= 1




print('\tpde:', pde)
print('\tpde discr.:', pdeD)
print('\tpde discr. with error(von Neumann/Fourier):', pdeDE)
print('\tchar eq:', pdeDEC)
print('\tchar. roots of difference eq:', roots  )
print('\tchar. roots simplified:', r_simple)
