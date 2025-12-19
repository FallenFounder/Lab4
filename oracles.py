import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        loss = np.mean(np.logaddexp(0, z))
        reg = self.regcoef / 2.0 * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        s = expit(z)  # sigma(z)
        grad_loss = -self.matvec_ATx(self.b * s) / self.b.size
        return grad_loss + self.regcoef * x

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        s = expit(z)
        w = s * (1 - s)
        hess_loss = self.matmat_ATsA(w) / self.b.size
        return hess_loss + self.regcoef * np.eye(x.size)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._cache_x = None
        self._cache_Ax = None
        self._cache_d = None
        self._cache_Ad = None

    def _update_cache_x(self, x):
        if self._cache_x is None or not np.array_equal(self._cache_x, x):
            self._cache_x = np.copy(x)
            self._cache_Ax = self.matvec_Ax(x)

    def _update_cache_d(self, d):
        if self._cache_d is None or not np.array_equal(self._cache_d, d):
            self._cache_d = np.copy(d)
            self._cache_Ad = self.matvec_Ax(d)

    def func(self, x):
        self._update_cache_x(x)
        Ax = self._cache_Ax
        z = -self.b * Ax
        loss = np.mean(np.logaddexp(0, z))
        reg = self.regcoef / 2.0 * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        self._update_cache_x(x)
        Ax = self._cache_Ax
        z = -self.b * Ax
        s = expit(z)
        grad_loss = -self.matvec_ATx(self.b * s) / self.b.size
        return grad_loss + self.regcoef * x

    def hess(self, x):
        self._update_cache_x(x)
        Ax = self._cache_Ax
        z = -self.b * Ax
        s = expit(z)
        w = s * (1 - s)
        hess_loss = self.matmat_ATsA(w) / self.b.size
        return hess_loss + self.regcoef * np.eye(x.size)

    def func_directional(self, x, d, alpha):
        self._update_cache_x(x)
        self._update_cache_d(d)
        Ax = self._cache_Ax
        Ad = self._cache_Ad
        x_alpha = x + alpha * d
        z = -self.b * (Ax + alpha * Ad)
        loss = np.mean(np.logaddexp(0, z))
        reg = self.regcoef / 2.0 * np.dot(x_alpha, x_alpha)
        return loss + reg

    def grad_directional(self, x, d, alpha):
        self._update_cache_x(x)
        self._update_cache_d(d)
        Ax = self._cache_Ax
        Ad = self._cache_Ad
        x_alpha = x + alpha * d
        z = -self.b * (Ax + alpha * Ad)
        s = expit(z)
        grad_loss_dir = -np.dot(self.b * s, Ad) / self.b.size
        grad_reg_dir = self.regcoef * np.dot(x_alpha, d)
        return grad_loss_dir + grad_reg_dir


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        if scipy.sparse.issparse(A):
            return A.T.dot(A.multiply(s.reshape(-1, 1)))
        return A.T.dot(A * s.reshape(-1, 1))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    result = np.zeros(n, dtype=float)
    f_x = func(x)
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        result[i] = (func(x + e) - f_x) / eps
    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    result = np.zeros((n, n), dtype=float)
    f_x = func(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_x_ei = func(x + e_i)
        for j in range(i, n):
            e_j = np.zeros(n)
            e_j[j] = eps
            f_x_ej = func(x + e_j)
            f_x_eiej = func(x + e_i + e_j)
            value = (f_x_eiej - f_x_ei - f_x_ej + f_x) / (eps ** 2)
            result[i, j] = value
            result[j, i] = value
    return result
