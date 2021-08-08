import logging
import multiprocessing
import numpy as np
from nltk.metrics.agreement import AnnotationTask

from xrr.distance_metrics import *

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
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S ', level=logging.INFO)

        if len(X[object_col].unique()) != len(Y[object_col].unique()):
            # TO DO: remove objects not judged by both if theres a difference
            valid_objects = np.intersect1d(X[object_col].unique(), Y[object_col].unique())
            X = X[X[object_col].isin(valid_objects)]
            Y = Y[Y[object_col].isin(valid_objects)]

        self.X = X.loc[:, [rater_col,object_col, answer_col]]
        self.Y = Y.loc[:, [rater_col,object_col, answer_col]]

        self.X_ = X.pivot_table(index=rater_col, columns=object_col, values=answer_col, aggfunc="first")
        self.Y_ = Y.pivot_table(index=rater_col, columns=object_col, values=answer_col, aggfunc="first")

        self.X_dict = self.X_.to_dict()
        self.Y_dict = self.Y_.to_dict()

        self.X_dict = {k: {k_:v_ for k_, v_ in v.items() if not np.isnan(v_)} for k,v in self.X_dict.items()}
        self.Y_dict = {k: {k_:v_ for k_, v_ in v.items() if not np.isnan(v_)} for k,v in self.Y_dict.items()}

        self.dict_keys = list(self.X_dict.keys())
        
        self.R = self.X_.count().values
        self.S = self.Y_.count().values

        self.R_bold = self.R.sum()
        self.S_bold = self.S.sum()

        self.rater_col = rater_col
        self.object_col = object_col
        self.dist_func = dist_func
        self.n = len(X[object_col].unique())
        self.kappa = None

    def kappa_x(self):
        """Cross kappa method. Indicates cross replication reliability between X and Y

        Returns:
            num: kappa_x value
        """
        if not self.kappa:
            self.kappa = 1-(self.d_o()/self.d_e())
        
        return self.kappa

    def normalized_kappa_x(self, irr="alpha"):
        """normalized cross kappa

        Args:
            irr (str, optional): desired IRR metric. Defaults to "alpha".

        Returns:
            num: normalized kappa_x value
        """
        irrs = self._IRR(self.X, irr)*self._IRR(self.Y, irr)
        if irrs <= 0:
            raise ValueError("Cannot normalize. IRR product is equal to or less than 0")
        if self.kappa:
            return self.kappa/np.sqrt(irrs)
        return self.kappa_x()/np.sqrt(irrs)

    def _IRR(self, A, irr):
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
        if irr.lower() not in ["alpha", "kappa", "pi"]:
            raise ValueError("Unknown IRR metric")
        if irr == "alpha":
            t = AnnotationTask(
                data=A.values
            )
            return t.alpha()
        else:
            # metrics for 2 raters only
            if A.shape[0] != 2:
                raise IndexError("Replication does not have 2 raters. Use alpha.")
            t = AnnotationTask(data=A.values)
            if irr == "kappa": 
                return t.kappa()
            elif irr == "pi":
                return t.pi()

    def d_o(self,workers=multiprocessing.cpu_count()):
        """observed disagreement

        Args:
            n (int): number of objects
            workers (int, optional): Number of workers for multiprocessing. Defaults to CPU count.

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

    def d_e(self,workers=multiprocessing.cpu_count()):
        """expected disagreement

        Args:
            workers (int, optional): Number of workers for multiprocessing. Defaults to CPU count.

        Returns:
            num: expected disagreement value
        """

        de = 0
        
        # try to initialize subsets of this list parallely??
        ijs = [(i,j) for i in range(len(self.R)) for j in range(len(self.S))]
        with multiprocessing.Pool(workers) as p:
            results = p.map(self.de_inner, ijs)
            de = sum(results)
        de /= (self.R_bold * self.S_bold)
        
        logging.info("expected disagreement: {}".format(de))

        return de

    def de_inner(self,a):
        """inner expected disagreement summation

        Args:
            a (tuple/list): [description]

        Returns:
            [type]: [description]
        """
        i = a[0]
        j = a[1]
        de_i = 0

        #to try: itertools product
        for r in self.X_dict[self.dict_keys[i]].values():
            for s in self.Y_dict[self.dict_keys[j]].values():
                de_i += self.dist_func(r,s)
        return de_i

    def do_inner(self,i):
        """inner observed disagreement summation

        Args:
            i (int): i

        Returns:
            num: summation
        """
        
        do_i = 0
        for r in self.X_dict[self.dict_keys[i]].values():
            for s in self.Y_dict[self.dict_keys[i]].values():
                do_i += self.dist_func(r,s)*(self.R[i] + self.S[i])/(self.R[i] * self.S[i])
        return do_i
