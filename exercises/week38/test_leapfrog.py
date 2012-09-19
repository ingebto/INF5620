# exercise 16
from numpy import *
from matplotlib.pyplot import *

def leapfrogSolver(I, a, b, T, dt, N):
    """ Solve u'(t) = -a(t)u(t) + b(t), u(0) = I.
    Uses Forward Euler to compute u(1)."""
    T = N*dt
    t = linspace(0, T, N+1)
    u = zeros(N+1)
    u[0] = I
    u[1] = u[0] + dt*(-a[0]*u[0] + b[0])
    for n in range(1, N):
        u[n+1] = u[n-1] + 2*dt*(-a[n]*u[n] + b[n])
    return u, t

def exact_solution(t, I, a, b, N):
    u_e = zeros(N+1)
    for i in range(N+1):
        u_e[i] = b[i]/a[i] + (I - (b[i]/a[i]))*exp(-a[i]*t[i])
    return u_e    

def explore(I, a, b, T, dt, N, makeplot=True):
    """
    Run a case with the solver, compute error measure,
    and plot the numerical and exact solutions (if makeplot=True).
    """
    u, t = leapfrogSolver(I, a, b, T, dt, N) # Numerical solution

    
    

    if makeplot:
        figure() # create new plot
        M = 1000
        a = ones(M+1)
        b = ones(M+1)
        t_e = linspace(0, T, M+1)
        u_e = exact_solution(t_e, I, a, b, M)
        plot(t, u, 'r--') # red dashes 
        plot(t_e, u_e, 'b-') # blue line for exact sol.
        legend(['numerical', 'exact'], 'best')
        xlabel('t')
        ylabel('u')
        title('Leapfrog, dt=%g' %  dt)
        savefig('lf_%g.eps' % dt)
        show()
  

def dt_experiments(I, T, dt, makeplot = True):
    for i in dt:
        N = int(round(T/i))
        a = ones(N+1)
        b = ones(N+1)
        explore(I, a, b, T, i, N, makeplot = True)

def main():
    #I, a, T, makeplot, dt_values = read_command_line()
    
    I = 0.1;
    dt = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01];
    T = 4;
    
    

    dt_experiments(I, T, dt, makeplot = True)

main()    



