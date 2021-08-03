import logging
import multiprocessing
import numpy as np
from nltk.metrics.agreement import AnnotationTask

from .distance_metrics import *

class xRR:
    """Cross-Replication Reliability Measure (xRR) class
    """

    def __init__(self, X, Y, answer_col, rater_col, object_col, dist_func=categorical):
        """xRR object
        X and Y need to have same column names and numeric answer_col (encoded if categorical)
        Can accomodate missing values.

        Args:
            X (pd.DataFrame): dataset one
            Y (pd.DataFrame): dataset two
            answer_col (str): column in X & Y with rating data
            rater_col (str): column in X & Y with responder data
            object_col (str): column in X & Y with assessed object data
            dist_func (str): Distance function. defaults to categorical
        
        """
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S %p', level=logging.DEBUG)

        if len(X[object_col].unique()) != len(Y[object_col].unique()):
            ## TO DO: remove objects not judged by both if theres a difference
            valid_objects = np.intersect1d(X[object_col].unique(), Y[object_col].unique())
            X = X[X[object_col].isin(valid_objects)]
            Y = Y[Y[object_col].isin(valid_objects)]

        self.X = X.loc[:,[rater_col,object_col,answer_col]]
        self.Y = Y.loc[:,[rater_col,object_col,answer_col]]

        self.X_ = X.pivot_table(index=rater_col, columns=object_col, values=answer_col, aggfunc="first")
        self.Y_ = Y.pivot_table(index=rater_col, columns=object_col, values=answer_col, aggfunc="first")

        self.X_dict = self.X_.to_dict()
        self.Y_dict = self.Y_.to_dict()

        self.dict_keys = list(self.Y_dict.keys())
        
        self.R = self.X_.count().values
        self.S = self.Y_.count().values

        self.R_bold = self.R.sum()
        self.S_bold = self.S.sum()

        self.dist_func = dist_func
        self.n = len(self.R) #equivalent to len(self.S)
        self.kappa = None


    def kappa_x(self):
        """Cross kappa method. Indicates cross replication reliability between X and Y

        Returns:
            num: observed disagreement value
        """
    
        self.kappa = 1-(self.d_o()/self.d_e())
        
        return self.kappa

    def normalized_kappa_x(self,irr="alpha"):
        """normalized cross kappa

        Args:
            irr (str, optional): desired IRR metric. Defaults to "alpha".

        Returns:
            num: normalized kappa value
        """
        if self.kappa:
            return self.kappa/np.sqrt(self.IRR(self.X, irr)*self.IRR(self.Y, irr))
        return self.kappa_x()/np.sqrt(self.IRR(self.X, irr)*self.IRR(self.Y, irr))
    
    def IRR(self, A, irr):
        """Inter-rater reliability
        Uses nltk agreement to get IRR

        Args:
            A (DataFrame): data in the usual format
            irr (str): IRR metric. Choose from ["alpha", "kappa", "pi", "s"].

        Raises:
            ValueError: if unknown metric given

        Returns:
            num: IRR value
        """
        t = AnnotationTask(data=A.values)
        if irr.lower() not in ["alpha", "kappa", "pi", "s"]:
            raise ValueError("Unknown IRR metric")
        if irr == "alpha":
            return t.alpha()
        elif irr == "kappa":
            return t.kappa()
        elif irr == "pi":
            return t.pi()
        elif irr == "s":
            return t.S()

    def d_o(self,workers=multiprocessing.cpu_count()-1):
        """observed disagreement

        Args:
            n (int): number of objects
            workers (int, optional): Number of workers for multiprocessing. Defaults to user CPU count -1.

        Returns:
            num: observed disagreement value
        """

        do = 0
        ints = [i for i in range(self.n)]
        with multiprocessing.Pool(workers) as p:
            results = p.map(self.do_inner, ints)
            do = sum(results)
        do /= (self.R_bold + self.S_bold)
        logging.info("observed disagreement: {}".format(do))
        return do

    def d_e(self,workers=multiprocessing.cpu_count()-1):
        """expected disagreement

        Args:
            workers (int, optional): Number of workers for multiprocessing. Defaults to user CPU count -1.

        Returns:
            num: expected disagreement value
        """

        d_e = 0
        ijs = [(i,j) for i in range(self.n) for j in range(self.n)]
        with multiprocessing.Pool(workers) as p:
            results = p.map(self.de_inner, ijs)
            d_e = sum(results)
        d_e /= (self.R_bold*self.S_bold)
        logging.info("expected disagreement: {}".format(d_e))
        return d_e

    def de_inner(self,a):
        """inner expected disagreement summation

        Args:
            a (tuple/list): [description]

        Returns:
            [type]: [description]
        """
        i = a[0]
        j = a[1]
        de = 0
        for r in self.X_dict[self.dict_keys[i]].values():
            for s in self.Y_dict[self.dict_keys[j]].values():
                de += self.dist_func(r,s)
        return de

    def do_inner(self,i):
        """inner observed disagreement summation

        Args:
            i (int): i

        Returns:
            num: summation
        """
        
        do = 0
        for r in self.X_dict[self.dict_keys[i]].values():
            for s in self.Y_dict[self.dict_keys[i]].values():
                do += self.dist_func(r,s)
        return do * (self.R[i]+self.S[i])/(self.R[i]*self.S[i])
