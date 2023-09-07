import numpy as np
import pandas as pd
import warnings

def cdot(l1, l2):
    """Convenience function for taking dot product of two arrays """ 
    return np.dot(safe_cast(l1), np.transpose(safe_cast(l2)))
    
    
def safe_cast(mat):
    """Check if matrix is compatible with the numpy array format and then cast it to that format if it has not already been."""
    if type(mat) != np.ndarray:
        out = np.array(mat)
        print(out.shape)
        try:
            len(out)
        except:
            raise RuntimeError("Data cannot be cast to numpy array.")
        else:
            return out
    else:
        return mat
        
        
def to_df(X, y):
    """Convert X and y into a joint dataframe then return it along with the column names of all independent variables."""
    df_indep = pd.DataFrame(X)
    df_dep = pd.DataFrame(y, columns=["Class"])
    return (df_indep.columns, pd.concat([df_indep, df_dep], axis=1))
    
    
def set_labels(y):
    """Check that there are exactly two class labels and then convert them to 0 and 1"""
    y = safe_cast(y)
    labels = list(set(y))
    if len(labels) != 2:
        raise RuntimeError("Only two-class datasets are currently supported")
    lbls = {}
    lbls[labels[0]] = 0
    lbls[labels[1]] = 1
    for ix in range(len(y)):
        y[ix] = lbls[y[ix]]
    return y
    
    
def symmetrise(mat):
    """Takes a matrix and returns its symmetric component."""
    mat = safe_cast(mat)
    mat_tr = np.transpose(mat)
    return (mat + mat_tr) / 2
    
    
def check_symmetry(self):
        """Makes sure any covariance matrix passed by a custom covariance estimator is a proper covariance estimator. Not intended for external use."""
        if self.cv_mat == None:
            warnings.warn("Do not use 'check_symmetry' until a covariance matrix has been estimated")
            return False
        else:
            self.cv_mat = safe_cast(self.cv_mat)
         
            for row in self.cv_mat:
                if len(self.cv_mat) != len(row):
                    warnings.warn("Covariance matrix must have an equal number of rows and columns. Resorting to default covariance estimator.")
                    return False
            for ix in range(len(self.cv_mat)):
                for jx in range(ix + 1, len(self.cv_mat[ix])):
                    if not np.allclose(self.cv_mat[ix][jx], self.cv_mat[jx][ix]):
                        warnings.warn("Covariance matrix is not symmetric. Resorting to default covariance estimator")
                        return False
            return True
