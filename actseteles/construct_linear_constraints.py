from .active_set_elitist_es import np, arrayize

############ V1
@arrayize
def constraints(x):
    return np.array([1 - xi for xi in x[:m]])


@arrayize
def grad_constraints(x):
    A = np.zeros((n, m))
    A[:m, :m] = np.eye(m)
    return A


def grad_constraints_transpose(x):
    return grad_constraints(x).T

########### V2
def unit_vector(j, n):
    v = np.zeros(n)
    v[j] = 1
    return v


def make_constraints_list(m, n):
    constraints_list = []
    grad_constraints_list = []
    for j in range(m):
        constraints_list.append(arrayize(lambda x, j=j : 1 - x[j]))  # avoid late binding
        grad_constraints_list.append(arrayize(lambda x, j=j: - unit_vector(j, n)))
    return constraints_list, grad_constraints_list
