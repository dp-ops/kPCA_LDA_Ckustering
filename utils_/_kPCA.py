import torch
from typing import Tuple
import warnings

class KPCA:
    def __init__(self, gamma=1, n_components = None, kernel_type ='rbf', degree=3, coef0=1, remove_zeros: bool = True):
        self.gamma = gamma
        self.remove_zeros = remove_zeros
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.degree = degree
        self.coef0 = coef0
        self.alphas = None
        self.X_train = None
        self.lambdas = None
        self.explained_var = None
        self._x_fit = None

    def _rbf_kernel(self, x:torch.tensor, y:torch.tensor = None) -> torch.tensor: #, gamma: float=1
        x1 = x.to(torch.float32)
        if y is None:
            dist = torch.cdist(x1, x1, p=2).pow(2)
        else:
            y1 = y.to(torch.float32)
            dist = torch.cdist(x1,y1,p=2).pow(2)
        
        return torch.exp(-self.gamma * dist)

    def _linear_kernel(self, x:torch.tensor, y=None) -> torch.tensor:
        x1 = x.to(torch.float32)
        if y is None: 
            return torch.matmul(x1, x1.T)
        else:
            y1 = y.to(torch.float32)
            return torch.matmul(x1, y1.T)
        
    def _poly_kernel(self, x:torch.tensor, y=None, degree=3, coef0=1) -> torch.tensor:
        x1 = x.to(torch.float32)
        if y is None:
            dot = torch.matmul(x1, x1.T)
        else:
            y1 = y.to(torch.float32)
            dot = torch.matmul(x1, y1.T)
        return(dot + coef0).pow(degree)

    def _check_n_components(self, n_features: int) -> None:
        if self.n_components is None:
            self.n_components = n_features - 1
        elif self.n_components > n_features:
            self.n_components = n_features
        elif 0 < self.n_components < 1:
            self.n_components = self._pov_to_n_components()
        elif 1<= self.n_components <= n_features:
            pass

        else:
            raise RuntimeError('The number of components should be between 1 and {}, '
                                                  'or between (0, 1) for the pov, '
                                                  'in order to choose the number of components automatically.\n'
                                                  'Got {} instead.'
                                                  .format(n_features, self.n_components))
        
        #keep explained var for n _components
        self.explained_var = self.explained_var[:self.n_components]

    @staticmethod 
    def _one_ns(size: int) -> torch.tensor:
        #creates a (shape, shape) symmetric matrix of all 1's divided by shape.

        return torch.ones((size, size)) / size
    
    @staticmethod 
    def _center_matrix(kernel_matrix: torch.Tensor) -> torch.Tensor: 
        """ Centers a matrix. :param kernel_matrix: the matrix to be centered. :return: the centered matrix. """ 
        # If kernel is 1D, which means that we only have 1 test sample, 
        # # expand its dimension in order to be 2D. 
        if kernel_matrix.ndimension() == 1: 
            return kernel_matrix.unsqueeze(0) 
        
        # Get the kernel's shape. 
        m, n = kernel_matrix.shape 
        # Create one matrices. 
        one_m = KPCA._one_ns(m) 
        one_n = KPCA._one_ns(n) 
        
        # Center the kernel matrix. 
        centered_matrix = kernel_matrix - one_m.matmul(kernel_matrix) - kernel_matrix.matmul(one_n) + one_m.matmul(kernel_matrix).matmul(one_n) 
        
        return centered_matrix
    
    @staticmethod 
    def _center_symmetric_matrix(kernel_matrix: torch.Tensor) -> torch.Tensor: 
        """ Centers a symmetric matrix. Slightly more efficient than the _center_matrix function. """
        # Get the kernel's shape. 
        n = kernel_matrix.shape[0] 
        # Create one n matrix. 
        one_n = KPCA._one_ns(n) 
        # Center the kernel matrix. 
        centered_matrix = kernel_matrix - one_n.matmul(kernel_matrix) - kernel_matrix.matmul(one_n) + one_n.matmul(kernel_matrix).matmul(one_n) 
        
        return centered_matrix

    def _pov_to_n_components(self) -> int:
            #pov: proportion o variance 
            pov = torch.cumsum(self.explained_var)
            nearest_value_index = (torch.abs(pov - self.n_components)).argmin()

            return nearest_value_index + 1

    def _clean_eigs(self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes the eigenvalues and eigenvectors and returns them cleaned."""
        #set a numerical tolerance for identifying valid eigenvalues
        tolerance = 1e-10

        #valid eigenvalues (positive or close to zero) 
        keep_indexes = torch.where(eigenvalues >= tolerance)[0]

        #warn if any eigenval is removed 
        if len(keep_indexes) < len(eigenvalues):
            warnings.warn(
                f"Negative or near-zero eigenvalues encountered and removed: {eigenvalues[eigenvalues < tolerance]}",
                RuntimeWarning
            )

        #keep only valid eigenvals and corresponding eigenvec
        eigenvalues = eigenvalues[keep_indexes]
        eigenvectors = eigenvectors[:, keep_indexes]

        #sort eigenval and eigenvec in descending order
        sorted_indexes = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indexes]
        eigenvectors = eigenvectors[:, sorted_indexes]

        return eigenvalues, eigenvectors



    def fit(self, x:torch.tensor) -> torch.tensor:

        self._x_fit = x

        if self.kernel_type == 'rbf':
           #passing the distances through the rbf kernel
            K_old = self._rbf_kernel(x) #, sq_dist  , gamma=self.gamma
        elif self.kernel_type == 'linear':
            K_old = self._linear_kernel(x)
        
        elif self.kernel_type == 'poly':
            K_old = self._poly_kernel(x, degree=self.degree, coef0=self.coef0)

        else:
            raise ValueError("Unsupported kernel. Try 'rbf', 'linear' or 'poly'.")

        #centering the kernel
        K = self._center_symmetric_matrix(K_old)

        eigenvals, eigenvecs = torch.linalg.eigh(K)
        self.lambdas, self.alphas = self._clean_eigs(eigenvals, eigenvecs)
        self.explained_var = self.lambdas / torch.sum(self.lambdas)
        self._check_n_components(self._x_fit.shape[1])
        self.alphas = self.alphas[:, :self.n_components]
        self.lambdas = self.lambdas[:self.n_components]

        return K
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        #projects the given data to the created feature space.

        if self.kernel_type == 'rbf':
           #passing the distances through the rbf kernel
            K_old = self._rbf_kernel(self._x_fit, x) #, sq_dist , gamma=self.gamma
        elif self.kernel_type == 'linear':
            K_old = self._linear_kernel(self._x_fit, x)
        
        elif self.kernel_type == 'poly':
            K_old = self._poly_kernel(self._x_fit, x, degree=self.degree, coef0=self.coef0)

        else:
            raise ValueError("Unsupported kernel. Try 'rbf', 'linear' or 'poly'.")
        
        # Center the kernel matrix.
        kernel_matrix = self._center_matrix(K_old)

        # Return the projected data.
        return kernel_matrix.T @ (self.alphas / torch.sqrt(self.lambdas))
