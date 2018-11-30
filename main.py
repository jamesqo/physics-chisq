#!/usr/bin/env python3

import numpy as np

def main():
    xs = [0.01291611793, 0.0080245764]
    ys = [29.224836, 32.844361]
    ss = [3.522054061, 0.06556646946]

    e11 = sum(x*x/(s*s) for x, s in zip(xs, ss))
    e12 = sum(x/(s*s) for x, s in zip(xs, ss))
    e21 = sum(x/(s*s) for x, s in zip(xs, ss))
    e22 = sum(1/(s*s) for x, s in zip(xs, ss))
    error = np.array([
        [e11, e12],
        [e21, e22]
    ])
    print(f"Error matrix shape: {error.shape}")
    cov = np.linalg.inv(error)
    print(f"Covariance matrix shape: {cov.shape}")

    rhs1 = sum(x*y/(s*s) for x, y, s in zip(xs, ys, ss))
    rhs2 = sum(y/(s*s) for y, s in zip(ys, ss))
    rhs = np.array([rhs1, rhs2])
    print(f"Right-hand side shape: {rhs.shape}")

    params = cov.dot(rhs)
    print(f"Parameter vector shape: {params.shape}")
    print(f"a = {params[0]}")
    print(f"b = {params[1]}")

if __name__ == '__main__':
    main()
