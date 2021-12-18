# Python implementation of the Active-Set (1+1)-ES

We implement the elitist evolution strategy with active-set constraint handling and a projection/repair method based on SciPy's SLSQP.

The active constraints are detected from the nonzero values of the Lagrange multipliers' estimates after SLSQP svolved the projection subproblem. To achieve this, we need to slightly modify the output of the SciPy's Python wrapping code around the core Fortran routine. See also [this issue](https://github.com/scipy/scipy/issues/14394).

## Use the algorithm

The Python code implements a `ActiveSetElitistES` class. The algorithm assume explicit constraints, which means the user must provide the constraints and constraints jacobian functions as callables to the class object. You can run the algorithm with an ask-and-tell interface (see also the `__main__` part of the python script for an example).

![plot_active_set_elitist_es_on_sphere.png]

## References

Arnold, D.V. (2016) *‘An Active-Set Evolution Strategy for Optimization with Known Constraints’*, in Handl, J. et al. (eds) Parallel Problem Solving from Nature – PPSN XIV. Cham: Springer International Publishing (Lecture Notes in Computer Science), pp. 192–202. doi:10.1007/978-3-319-45823-6_18.

This is work in progress, in the future I plan to implement the feature following from this paper:

Arnold, D.V. (2017) *‘Reconsidering constraint release for active-set evolution strategies’*, in Proceedings of the Genetic and Evolutionary Computation Conference. GECCO ’17: Genetic and Evolutionary Computation Conference, Berlin Germany: ACM, pp. 665–672. doi:10.1145/3071178.3071294.
# active_set_es
