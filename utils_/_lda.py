import torch
import warnings
from typing import Tuple
from typing import Union 

class LdaNotFeasibleException(Exception):
    pass

class OneSamplePassedException(Exception):
    pass

class InvalidNumOfComponentsException(Exception):
    pass

class NotFittedException(Exception):
    pass

class LDA():
    def __init__(self, n_components: Union[int, float] = None, remove_zeros: bool = True):
        self.n_components = n_components
        self.remove_zeros = remove_zeros
        self.explained_var = None
        self.labels = None
        self.lebels_counts = None
        self._n_classes = None
        self._n_features = None
        self._w = None

    def _check_if_LDA_possible(self, x: torch.tensor) -> None:
        #check if lda is possible
        n_samples = x.shape[0]

        if n_samples < 2:
            raise OneSamplePassedException('Cant perform LDA for 1 sample')

        if n_samples < self._n_features:
            raise LdaNotFeasibleException('LDA is not feasible if the number of samples is less than the number of features.' 
                                          ' You seem to have {} samples and {} features.' 
                                          .format(n_samples, self._n_features))
        
    def __set_state(self, x: torch.Tensor, y: torch.Tensor) -> None:
        #get labels and number of instances of each class
        self._labels, self._labels_count = torch.unique(y, return_counts = True)
        #get num of classes
        self._n_classes = len(self._labels)
        #get num of features
        self._n_features = x.shape[1]

    def _pov_to_n_components(self) -> int:
        #Finds the number of components needed in order to achieve the given pov.
        #proportion of variance = pov
        pov = torch.cumsum(self.explained_var) ###, dim=0
        #index of the nearest poc value to given pov preference
        nearest_value_index = (torch.abs(pov-self.n_components)).argmin()
        
        return nearest_value_index

    def _check_n_components(self) -> None:
        #if n_components not passed, return num of classes -1 
        if self.n_components is None:
            self.n_components = self._n_classes -1 
        #if n_components > n_classes use n_classes -1 
        elif self.n_components >= self._n_classes:
            self.n_components = self._n_classes -1
        #if n components have been given a correct value, pass
        elif 1 <= self.n_components < self._n_classes:
            pass
        #if pov has been passed, return as many n_components as needed
        elif 0 < self.n_components < 1:
            self.n_components = self._pov_to_n_components()
        
        else:
            raise InvalidNumOfComponentsException('The number of components should be between 1 and {}, ' 
                                                  'or between (0, 1) for the pov, ' 
                                                  'in order to choose the number of components automatically.\n' 
                                                  'Got {} instead.' .format(self._n_classes, self.n_components))

    def _class_means(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        #calculate the mean of every feature for each class.
        means = torch.zeros((self._n_classes, self._n_features), dtype=torch.float64)

        for c, label in enumerate(self._labels):
            means[c] = torch.mean(x[y==label], dim=0, dtype=torch.float64)

        return means
    
    def _sw(self, x: torch.tensor, y: torch.tensor, class_means: torch.tensor) -> torch.tensor:
        #within scatter matrix
        #instantiate an array for the within class scatter matrix.
        sw = torch.zeros((self._n_features, self._n_features))

        for label_index, label in enumerate(self._labels):
            #create the Si
            si = torch.zeros((self._n_features, self._n_features))
            group_samples = x[y == label]
            n_group_samples = group_samples.shape[0]

            #now for every sample
            for sample, sample_index in zip(group_samples, range(n_group_samples)):
                #difference of the sample vector from the mean vector
                diff = sample - class_means[label_index]
                #expland dimentions
                diff = diff.unsqueeze(1)
                #sum the dot product of diff with its transposed self
                si += diff.mm(diff.T)

            # sum the si to create the Scatter within each class matrix
            sw += si
        return sw
    
    def _sb(self, class_means: torch.tensor, x_mean: torch.tensor) -> torch.tensor:
        #calculates the between class scatter matrix.
        sb = torch.zeros((self._n_features, self._n_features))
        
        for mean_vec, count in zip(class_means, self._labels_count):
            # Convert mean vector to a column vector
            mean_vec_col = mean_vec.unsqueeze(1)
            diff = mean_vec_col - x_mean
            # Multiply the number of the class instances
            # with the dot product of the difference with itself's transpose
            # and sum the result to the Sb matrix.
            sb+= count * diff.mm(diff.T)  #torch.dot(diff, diff.T)

        return sb

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
    
    def fit(self, x:torch.tensor, y:torch.tensor, n_comp = None) -> torch.tensor:
        if n_comp is None:
            pass
        elif n_comp > self.n_components:
            raise ValueError("The number of components should be the same or less of the sum of classes in the data")
        else:
            self.n_components = n_comp
        
        #fit the lda model

        self.__set_state(x, y)
        self._check_if_LDA_possible(x)

        #calc the x means and class means
        x_mean = x.mean(axis = 0, dtype = torch.float64) 
        class_means = self._class_means(x, y)

        #get the scatter matrixes
        sw = self._sw(x, y, class_means)
        sb = self._sb(class_means, x_mean.unsqueeze(1))
        sw_inv_sb = torch.matmul(torch.linalg.inv(sw), sb)

        #get eigencalues and eigenvectors
        eigenvals, eigenvec = torch.linalg.eigh(sw_inv_sb)
        #process the eigenvalues and eigenvectors.
        eigenvalues, eigenvectors = self._clean_eigs(eigenvals, eigenvec)
        
        #TODO check if the correct eigenvects are selected
        self.explained_var = torch.divide(eigenvalues, torch.sum(eigenvectors))
        self._check_n_components()

        #store the needed eigenvectors
        self._w = eigenvectors[: , :self.n_components]

        return self._w
    
    def transform(self, x: torch.tensor) -> torch.tensor:
        if self._w is None:
            raise NotFittedException('KPCA has not been fitted yet!')
        
        return torch.mm(x, self._w)
    
    def get_params(self) -> dict:
        #return the ldas params
        params = dict(n_components = self._param_values(self.n_components),
                      renove_zeros = self.remove_zeros)
        
        return params
