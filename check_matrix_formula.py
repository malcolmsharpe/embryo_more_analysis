import itertools
import numpy as np
from scipy.misc import factorial

n = 5
A = np.random.randn(n,n)

brute_sum = 0
for sigma in itertools.permutations(range(n)):
    term = 0
    for i in range(n):
        term += A[i][sigma[i]]
    brute_sum += term**2

def norm2(x):
    return (x**2).sum()

clever_sum = factorial(n-2) * (n * norm2(A) + A.sum()**2 - norm2(A.dot(np.ones(n))) - norm2(A.transpose().dot(np.ones(n))))

numpy_sum = factorial(n-2) * (n * np.linalg.norm(A)**2 + A.sum()**2 - np.linalg.norm(A.dot(np.ones(n)))**2 - np.linalg.norm(A.transpose().dot(np.ones(n)))**2)

print(f'Brute sum  = {brute_sum:.6f}')
print(f'Clever sum = {clever_sum:.6f}')
print(f'Numpy sum  = {numpy_sum:.6f}')
print()
print('Clever expression =')
print(f'  {factorial(n-2)} * (')
print(f'    {n * norm2(A)}')
print(f'  + {A.sum()**2}')
print(f'  - {norm2(A.dot(np.ones(n)))}')
print(f'  - {norm2(A.transpose().dot(np.ones(n)))} )')
