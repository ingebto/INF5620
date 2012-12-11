from dolfin import *
import numpy as np

# Create mesh
mesh = UnitCube(8, 8, 8)

# Define a MeshFunction over two subdomains
subdomains = MeshFunction('uint', mesh, 3)

class Omega0(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] <= 0.5 else False

class Omega1(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] >= 0.5 else False
# note: it is essential to use <= and >= in the comparisons

# Initilize and mark subdomains 
subdomain0 = Omega0()
subdomain0.mark(subdomains, 0)
subdomain1 = Omega1()
subdomain1.mark(subdomains, 1)

V0 = FunctionSpace(mesh, 'DG', 0)
E, nu = Function(V0), Function(V0)
print len(E.vector().array())

print 'mesh:', mesh
print 'subdomains:', subdomains
print "E: ", E
print "nu: ", nu 

# Loop over cells, extract corresponding 
# subdomain number, and fill cell value in mu, lmbda
E_values = [1.0, 5000.0]
nu_values = [0.1, 100.5]
print "len(subdomains.array()):", len(subdomains.array())
for cell_no in range(len(subdomains.array())):
	# print cell_no
	subdomain_no = subdomains.array()[cell_no]
	E.vector()[cell_no] = E_values[subdomain_no]
	nu.vector()[cell_no] = nu_values[subdomain_no]

print "E degree of freedom: ", E.vector().array()
print "len(E)", len(E.vector().array())
print "nu degree of freedom: ", nu.vector().array()
print "len(nu)", len(nu.vector().array())

# plot(subdomains, title = "subdomains") 
# interactive()

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

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

# Lame's Elasticity parameters and other constants
E, nu = 1.0, 0.1
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
rho = 1.0
alpha = 1.0
gamma = -(lmbda*alpha)/(2*mu + 2*lmbda)

c = Expression(("0.0", "gamma*x[1]", "gamma*x[2]"), gamma = gamma)
r = Expression(("alpha*x[0]", "gamma*x[1]", "gamma*x[2]"), alpha = alpha, gamma = gamma)
bc_l = DirichletBC(V, c, left_boundary)
bc_r = DirichletBC(V, r, right_boundary)
bcs = [bc_l, bc_r]


# Define variational problem
u, v = TrialFunction(V), TestFunction(V)
n = FacetNormal(mesh)
b = Constant((0.0, 0.0, 0.0))

# Governing equation
F = inner(sigma(u), grad(v))*dx - rho*dot(b, v)*dx # - dot(dot(sigma(u), n), v)*ds
a, L = lhs(F), rhs(F)

# Set up PDE and solve
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bcs=bcs)
solver = LinearVariationalSolver(problem)
solver.solve()

exact = Expression(("alpha*x[0]", "gamma*x[1]", "gamma*x[2]"), alpha=alpha, gamma=gamma)
u_e = interpolate(exact, V)
error = np.max(u_e.vector().array() - u.vector().array())
print "\nerror", error


plot(u, mode="displacement", axes=True)
# plot(u, mode="displacement", axes=True)
plot(mesh)
interactive()
