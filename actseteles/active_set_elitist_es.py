"""
"""
from functools import wraps
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp


def arrayize(func):
    """
    Sanitize input as NumPy array
    """
    @wraps(func)  # to propagate __name__
    def wrapper(x):
        x = np.asarray(x, dtype=np.float64)
        return func(x)
    return wrapper


@arrayize
def sphere(x):
    return np.sum(x**2)


@arrayize
def grad_sphere(x):
    return 2 * x


class ActiveSetElitistES:
    """
    """

    def __init__(
        self, x0, fx0, sigma0, constraints_list, grad_constraints_list,
        logging=True, display=True):
        """Ask-and-tell interface for the active-set (1+1)-ES

        constraints_list is a list of callable constraint functions
        same for grad_constraints_list
        """
        self.x0 = x0
        self.constraints_list = constraints_list
        self.grad_constraints_list = grad_constraints_list
        self.J = set(range(self.m))
        self.active_set = set()
        self.eqset = set()

        # Adaptive parameters
        self.x = x0
        self.fx = fx0
        self.fx_ = np.inf
        self.sigma = sigma0

        # Hyperparameters
        self.p = .2
        self.eps_cons = 1e-8

        # SLSQP parameters
        self.slsqp_options = {
            "iter": 100 * self.dimension,
            "acc": 1e-6,  # as in the paper for fmincon
                          # but is it the same meaning ?
                          # is self.sigma / 10,  # it this a good idea ?
            "full_output": True,
            "disp": False,
        }

        # Stopping parameters
        self.iteration_counter = 0
        self.tolfun = 1e-11
        self.maxiter = 500

        self.display = display
        self.logging = logging
        if self.log:
            self.col_names = ["N_iter", "sigma", "x", "fx", "kappa", "j", "n_"]
            self.rundata = pd.DataFrame(columns=self.col_names)
            self.rundata.set_index("N_iter")

    @property
    def dimension(self):
        return len(self.x0)

    @property
    def m(self):
        return len(self.constraints_list)

    @property
    def m_eq(self):
        return len(self.eqset)

    def constraints(self, x):
        """Returns the vector of m constraints values at x"""
        return np.asarray(
            [c(x) for c in self.constraints_list])

    def grad_constraints(self, x):
        """Returns the Jacobian matrix with m row and n columns"""
        return np.asarray(
            [gc(x) for gc in self.grad_constraints_list])

    def eqconstraints(self, w):
        """Equality constraints"""
        return np.array([self.constraints_list[j](w) for j in self.eqset])

    def ieqconstraints(self, w):
        """Inequality constraints

        NB: inactive constraints are passed as inequality constraints
        which are required >= 0 in the slsqp
        """
        return np.array(
            [- self.constraints_list[j](w) for j in self.J - self.eqset])

    def grad_eqc(self, w):
        """Jacobian of equality constraints"""
        return np.array([self.grad_constraints_list[j](w) for j in self.eqset])

    def grad_ieqc(self, w):
        """Jacobian of inequality constraints

        NB: inactive constraints are passed as inequality constraints
        which are required >= 0 in the slsqp
        """
        return np.array(
            [- self.grad_constraints_list[j](w) for j in self.J - self.eqset])

    def project(self, y, reduced_search_space=False):
        """
        Projection of y onto the feasible set, solving an explicit LCQP

        If reduced_search_space is true, then project onto the feasible set
        constrained to equality in the current active set
        Solved with SciPy's SLSQP routine
        (wrapper around Dieter Kraft's Fortran code)
        """
        def distance(w):
            return sphere(w - y)

        def grad_distance(w):
            return grad_sphere(w - y)

        if reduced_search_space:
            self.eqset = self.active_set
        else:
            self.eqset = set()

        res = fmin_slsqp(
            distance, self.x0,
            f_eqcons=self.eqconstraints if self.m_eq > 0 else None,
            f_ieqcons=self.ieqconstraints if self.m_eq < self.m else None,
            fprime=grad_distance,
            fprime_eqcons=self.grad_eqc if self.m_eq > 0 else None,
            fprime_ieqcons=self.grad_ieqc if self.m_eq < self.m else None,
            **self.slsqp_options
        )
        if res.status != 0:
            warnings.warn(f"SLSQP exit with {res.message}")
        return res

    def ask(self):
        # Check reduced subspace
        A = self.grad_constraints(self.x).T  # now n rows, m columns
        A = A[:, list(self.active_set)]
        if A.size > 0:
            self.n_ = self.dimension - np.linalg.matrix_rank(A)
        else:
            self.n_ = self.dimension

        j = 0
        while True:
            # Sample
            z = np.random.randn(self.dimension)
            y = self.x + self.sigma * z
            # Project
            if self.n_ == 0:
                self.kappa = True
            else:
                p_ = np.random.uniform()
                if p_ < self.p:
                    self.kappa = True
                else:
                    self.kappa = False
            res = self.project(y, not self.kappa)  # why is that so ?
            y_proj = res.x
            gy = self.constraints(y_proj)
            self.active_set = set(np.where(res.lag > 0)[0])
            j += 1
            if np.all(gy <= 0 + self.eps_cons):
                break
        self.number_of_inner_loops = j

        return y_proj

    def tell(self, y, fy):
        # Incumbent update and one-fifth rule step size adaptation
        if fy < self.fx:
            self.x = y
            self.fx_ = self.fx
            self.fx = fy
            if not self.kappa:
                self.sigma *= 2**(1 / self.n_)
        elif not self.kappa:
            self.sigma *= 2**(-1 / 4 / self.n_)
        self.iteration_counter += 1
        self.disp()
        self.log()

    def disp(self):
        if self.display:
            if not self.iteration_counter % 50:
                print(f"{self.iteration_counter:<10} {self.sigma:<10.3e} \
                      {self.fx:>20.3e}")

    def log(self):
        if self.logging:
            self.rundata = self.rundata.append(
                pd.Series(
                    (self.iteration_counter,
                     self.sigma,
                     self.x,
                     self.fx,
                     self.kappa,
                     self.number_of_inner_loops,
                     self.n_),
                    index=self.col_names),
                ignore_index=True)

    def stop(self):
        stopdict = {}
        if self.iteration_counter > self.maxiter:
            stopdict["maxiter"] = self.maxiter
        dfx = self.fx_ - self.fx
        if dfx < self.tolfun:
            stopdict["tolfun"] = dfx
        return stopdict

    def plot(self):
        if not self.log:
            raise ValueError("Nothing to plot")

        index_suspended = np.where(
            np.logical_and(
                self.rundata["n_"] > 0, self.rundata["kappa"] == True))

        plot_options = {
            "lw": 1,
        }

        plt.semilogy(self.rundata.index, self.rundata["fx"] - self.m,
                     c="blue", label="$f(x)$", **plot_options)
        plt.semilogy(self.rundata.index, self.rundata["sigma"],
                     c="orange", label="$\sigma$", **plot_options)
        plt.scatter(index_suspended,
                    self.rundata["fx"].iloc[index_suspended] - self.m,
                    marker="o", s=10, facecolors="none", edgecolors="orange",
                    lw=.8, label="$n' >0, \kappa = 1$")
        plt.xlabel("Iteration number")
        plt.grid(True)
        plt.legend()
        plt.title(f"Trajectory of the active-set (1+1)-ES\
                  \nSphere problem, n={self.dimension}, m={self.m}")
        plt.savefig("figures/plot_active_set_elitist_es_on_sphere.pdf", format="pdf")


if __name__ == "__main__":
    from .construct_linear_constraints import make_constraints_list

    dimension = 10
    m = 5

    cl, gcl = make_constraints_list(m, dimension)
    x0 = np.ones(dimension) * dimension
    es = ActiveSetElitistES(x0, sphere(x0), 1, cl, gcl)

    while not es.stop():
        y = es.ask()
        fy = sphere(y)
        es.tell(y, fy)

    es.plot()
