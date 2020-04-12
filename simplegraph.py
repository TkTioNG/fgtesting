import numpy as np
import factorgraph as fg


g = fg.Graph()


g.rv('a', 2)
g.rv('b', 3)
g.rv('c', 2)
g.rv('d', 3)


g.factor(['a'], potential=np.array([0.3, 0.7]))
g.factor(['b', 'a'], potential=np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9]
]))

g.factor(['c'], potential=np.array([0.3, 0.7]))
g.factor(['c', 'd'], potential=np.array([
        [0.2, 0.4, 0.1],
        [0.8, 0.6, 0.9]
]))

g.rv('s', 5)
g.rv('y', 2)

#[0.672,0.473,0.1,0.897,0.587]
#[0.24624405, 0.17332356, 0.03664346, 0.32869183, 0.21509711]
g.factor(['s'], potential=np.array([0.672,0.473,0.1,0.897,0.587]))

g.factor(['s','y'], potential=np.array([
        [0.3, 0.7],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.7, 0.3],
        [0.6, 0.4],
        ]))

iters, converged = g.lbp(normalize=True)
print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
print()


g.print_messages()
print()


g.print_rv_marginals()


