import sympy
from sympy import Matrix, pprint
from sympy.printing import pycode

if __name__ == "__main__":
    alpha, beta, gamma = sympy.symbols('alpha beta gamma')

    x, y, z = sympy.symbols('x y z')

    r1 = Matrix([[1, 0, 0],
                 [0, sympy.cos(gamma), -sympy.sin(gamma)],
                 [0, sympy.sin(gamma), sympy.cos(gamma)]])
    r2 = Matrix([[sympy.cos(beta), 0, sympy.sin(beta)],
                 [0, 1, 0],
                 [-sympy.sin(beta), 0, sympy.cos(beta)]])
    r3 = Matrix([[sympy.cos(alpha), -sympy.sin(alpha), 0],
                 [sympy.sin(alpha), sympy.cos(alpha), 0],
                 [0, 0, 1]])
    X = Matrix([[x, y, z]]).T

    x1, y1, z1 = (r3 * r2 * r1) * X

    pprint(x1)
    pprint(y1)
    pprint(z1)
    pprint(f"x1 = {pycode(x1)}")
    pprint(f"y1 = {pycode(y1)}")
    pprint(f"z1 = {pycode(z1)}")
