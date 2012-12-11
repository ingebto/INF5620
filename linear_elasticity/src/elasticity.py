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
alpha = 1.0
gamma = -(lmbda*alpha)/(2*mu + 2*lmbda)
rho = 1.0
dt = 0.1

# Create test and trial functions
u, v = TrialFunction(V), TestFunction(V)
n = FacetNormal(mesh)
b = Expression(("2.0*alpha*x[0]", "2.0*gamma*x[1]", "2.0*gamma*x[2]"), alpha=alpha, gamma=gamma)
u0 = Constant((0.0, 0.0, 0.0))
v0 = Constant((0.0, 0.0, 0.0))

def  epsilon(u):
	""" Strain tensor """
	return 0.5*(nabla_grad(u) + transpose(nabla_grad(u)))

def sigma(u):
	""" Stress tensor """
	return (2.0*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(v.cell().d))

# Dirichlet boundary conditions on left boundary
def left_boundary(x, on_boundary):
	tol = 1E-14
	return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
	tol = 1E-14
	return on_boundary and abs(x[0] - 1) < tol

def update_bc(t=0.0):
	# u = [t^2*alpha*x, t^2*gamma*y, t^2*gamma*z]^T
	c = Expression(("0.0", "t*t*gamma*x[1]", "t*t*gamma*x[2]"), gamma = gamma, t=t)
	r = Expression(("t*t*alpha*x[0]", "t*t*gamma*x[1]", "t*t*gamma*x[2]"), alpha = alpha, gamma = gamma, t=t)
	bc_l = DirichletBC(V, c, left_boundary)
	bc_r = DirichletBC(V, r, right_boundary)
	bcs = [bc_l, bc_r]
	return bcs

def exact(t=0.0):
	u_exact = Expression(("t*t*alpha*x[0]", "t*t*gamma*x[1]", "t*t*gamma*x[2]"), t=t, alpha=alpha, gamma=gamma)
	return u_exact

u_2 = interpolate(u0, V)
u_1 = TrialFunction(V)

# Gorverning equation for the first time step
F = dot(u_1, v)*dx - dot((u_2 + dt*v0 + (dt**2/2.0)*b), v)*dx \
		+ (dt**2/(2.0*rho))*inner(sigma(u_2), nabla_grad(v))*dx \
		- (dt**2/(2.0*rho))*dot(dot(sigma(u_2), n), v)*ds
a, L = lhs(F), rhs(F)

# Set up PDE for first time step and solve
u_1 = Function(V)
bcs = update_bc(dt)
problem = LinearVariationalProblem(a, L, u_1, bcs=bcs)
solver = LinearVariationalSolver(problem)
solver.solve()
# plot(u_1, mode="displacement", axes=True, title="t = %g" %dt)
# interactive()

# Error after the first time step
u_e = interpolate(exact(dt), V)
error = np.max(u_e.vector().array() - u_1.vector().array())
print "\nerror after the first time step:", error

# Governing equation for all other time steps
F = dot(u, v)*dx - (dot((2*u_1 - u_2 + dt**2*b), v) \
		+ (dt**2/rho)*inner(sigma(u_1), nabla_grad(v)))*dx \
		- (dt**2/rho)*dot(dot(sigma(u_1), n), v)*ds
a, L = lhs(F), rhs(F)
u = Function(V)

T = 1.5
t = 2*dt
while t <= T:
	bcs = update_bc(t=t)

	# Set up PDE and solve
	problem = LinearVariationalProblem(a, L, u, bcs=bcs)
	solver = LinearVariationalSolver(problem)
	solver.solve()
	
	# Error after time step t
	u_e = interpolate(exact(t), V)
	error = np.max(u_e.vector().array() - u.vector().array())
	print "\nerror after time t = %g:" %t, error

	# plot(u, mode="displacement", axes=True, title="t = %g" %t)
	# interactive()
	
	t += dt
	u_2.assign(u_1)
	u_1.assign(u)

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
print "t=%g sigma: \n" %t,matrix



