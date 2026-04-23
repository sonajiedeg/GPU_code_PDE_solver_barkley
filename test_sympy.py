import sympy as sp



N = sp.symbols('N', real=True)
expr = (sp.exp(2*sp.I*N) - 1)**2/((sp.exp(2*sp.I*N) - 1)**2 - 4*sp.exp(2*sp.I*N)*sp.cos(N)**2)
result = [a.trigsimp() for a in expr.rewrite(sp.cos).expand().as_real_imag()]
print(result)
