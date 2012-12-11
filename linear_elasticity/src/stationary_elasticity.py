from dolfin import *
import numpy as np

# Create mesh
elem = 8
mesh = UnitCube(elem, elem, elem)

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Lame's Elasticity parameters and other constants
E, nu = 1.0, 0.1
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
rho = 1.0
alpha = 1.0
gamma = -(lmbda*alpha)/(2*mu + 2*lmbda)

# Create test and trial functions
u, v = TrialFunction(V), TestFunction(V)
n = FacetNormal(mesh)
b = Constant((0.0, 0.0, 0.0))

def  epsilon(u):
	""" Strain tensor """
	#return 0.5*(nabla_grad(u) + transpose(nabla_grad(u)))
	return 0.5*(grad(u) + transpose(grad(u)))

def sigma(u):
	""" Stress tensor """
	return (2*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(v.cell().d))

def left_boundary(x, on_boundary):
	tol = 1E-14
	return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
	tol = 1E-14
	return on_boundary and abs(x[0] - 1) < tol

# Governing equation
F = inner(sigma(u), grad(v))*dx - rho*dot(b, v)*dx - dot(dot(sigma(u), n), v)*ds
a, L = lhs(F), rhs(F)

# u = [alpha*x, gamma*y, gamma*z]^T
c = Expression(("0.0", "gamma*x[1]", "gamma*x[2]"), gamma = gamma)
r = Expression(("alpha*x[0]", "gamma*x[1]", "gamma*x[2]"), alpha = alpha, gamma = gamma)
bc_l = DirichletBC(V, c, left_boundary)
bc_r = DirichletBC(V, r, right_boundary)
bcs = [bc_l, bc_r]

# Set up PDE and solve
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bcs=bcs)
solver = LinearVariationalSolver(problem)
solver.solve()

exact = Expression(("alpha*x[0]", "gamma*x[1]", "gamma*x[2]"), alpha=alpha, gamma=gamma)
u_e = interpolate(exact, V)
error = np.max(u_e.vector().array() - u.vector().array())
print "\nerror", error

V_ = TensorFunctionSpace(mesh, "Lagrange", 1)
sigma_ = project(sigma(u),V_)
epsilon_ = project(epsilon(u), V_)
matrix = np.zeros((3,3))
q = (elem+1)**3
matrix[0, 0] = sigma_.vector().array()[0*q]
matrix[0, 1] = sigma_.vector().array()[1*q]
matrix[0, 2] = sigma_.vector().array()[2*q]
matrix[1, 0] = sigma_.vector().array()[3*q]
matrix[1, 1] = sigma_.vector().array()[4*q]
matrix[1, 2] = sigma_.vector().array()[5*q]
matrix[2, 0] = sigma_.vector().array()[6*q]
matrix[2, 1] = sigma_.vector().array()[7*q]
matrix[2, 2] = sigma_.vector().array()[8*q]
print "alpha:", alpha
print "sigma: \n",matrix
#print len(sigma_.vector().array())

plot(u, mode="displacement", axes=True)
plot(mesh)
interactive()


