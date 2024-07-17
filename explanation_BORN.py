
import scipy.sparse
import numpy 
def explain(X=None, sample_weight=None):

    if X is None:
        return _weights()

    X = _normalize(X, axis=1)
    X = _power(X, 0.5)

    return _multiply(_weights(), _sum(X, axis=0).T)


def _dense():
    return  numpy

def _sparse():
    return scipy.sparse


def _weights(Pjk, a, b):
    P_jk = Pjk
    if b != 0:
        P_jk = _multiply(P_jk, _power(_sum(Pjk, axis=0), -b))
    if b != 1:
        P_jk = _multiply(P_jk, _power(_sum(Pjk, axis=1), b-1))

    W_jk = _power(P_jk, a)
    P_jk = _normalize(P_jk, axis=1)
    H_j = 1 + _sum(_multiply(P_jk, _log(P_jk)), axis=1) / _dense().log(P_jk.shape[1])
    W_jk = _multiply(W_jk, _power(H_j, 1))

    return W_jk, H_j, P_jk

# def _weights(Pjk):
#     P_jk = Pjk
#     P_jk = _multiply(P_jk, _power(_sum(Pjk, axis=0), -1))

#     W_jk = _power(P_jk, a)
#     P_jk = _normalize(P_jk, axis=1)
#     H_j = 1 + _sum(_multiply(P_jk, _log(P_jk)), axis=1) / _dense().log(P_jk.shape[1])
#     W_jk = _multiply(W_jk, _power(H_j, 1))

#     return W_jk, H_j, P_jk

def _sum(x, axis):
    if _sparse().issparse(x):
        return x.sum(axis=axis)

    return _dense().asarray(x).sum(axis=axis, keepdims=True)

def _multiply(x, y):
    if _sparse().issparse(x):
        return x.multiply(y).tocsr()

    if _sparse().issparse(y):
        return y.multiply(x).tocsr()

    return _dense().multiply(x, y)

def _power(x, p):
    x = x.copy()

    if _sparse().issparse(x):
        x.data = _dense().power(x.data, p)

    else:
        nz = _dense().nonzero(x)
        x[nz] = _dense().power(x[nz], p)

    return x

def _log(x):
    x = x.copy()

    if _sparse().issparse(x):
        x.data = _dense().log(x.data)

    else:
        nz = _dense().nonzero(x)
        x[nz] = _dense().log(x[nz])

    return x

def _normalize(x, axis, p=1.):
    s = _sum(x, axis)
    n = _power(s, -p)

    return _multiply(x, n)
