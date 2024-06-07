"""
@authors:
Collin Leiber
"""

from typing import Tuple
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


def _orthogonal_clustering(
    X: np.ndarray,
    n_clusters: list,
    explained_variance_for_clustering: float,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, list, list, list, np.ndarray]:
    """
    Start the actual Orthogonal clustering (Orth1) procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : list
        list containing number of clusters for each subspace
    explained_variance_for_clustering : float
        Defines the variances that is contained in the subspace used for clustering. If this value is 1, PCA will not be executed before performing KMeans
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, list, list, list, np.ndarray)
        The labels,
        The cluster centers,
        The projections,
        The PCA transformations,
        The mean value of the data set
    """
    assert (
        explained_variance_for_clustering > 0 and explained_variance_for_clustering <= 1
    ), "explained_variancefor_clustering must be within (0, 1)"
    labels = np.zeros((X.shape[0], len(n_clusters)), dtype=np.int32)
    cluster_centers = []
    projections = []
    pcas = [] if explained_variance_for_clustering != 1 else None
    # Center data
    global_mean = np.mean(X, axis=0)
    X = X - global_mean
    for subspace, k in enumerate(n_clusters):
        # (Optional) Execute PCA before clustering
        if explained_variance_for_clustering != 1:
            pca = PCA(explained_variance_for_clustering)
            X_subspace = pca.fit_transform(X)
            pcas.append(pca)
        else:
            X_subspace = X
        # Execute clustering
        km = KMeans(k, random_state=random_state)
        km.fit(X_subspace)
        # Save labels
        labels[:, subspace] = km.labels_
        # Get orthogonal space. Note that due to PCA, KMeans centers can be lower-dimensional
        centers_subspace = np.zeros((k, X.shape[1]))
        projections_subspace = []
        for c in range(k):
            # Get full-dimensional center
            center = np.mean(X[km.labels_ == c], axis=0)
            proj = np.identity(X.shape[1]) - center.reshape(-1, 1) @ center.reshape(
                1, -1
            ) / (center.reshape(1, -1) @ center.reshape(-1, 1))
            X[km.labels_ == c] = X[km.labels_ == c] @ proj
            centers_subspace[c] = center
            projections_subspace.append(proj)
        cluster_centers.append(centers_subspace)
        projections.append(projections_subspace)
    return labels, cluster_centers, projections, pcas, global_mean


def _clustering_in_orthogonal_spaces(
    X: np.ndarray,
    n_clusters: list,
    explained_variance_for_clustering: float,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, list, list, list, np.ndarray]:
    """
    Start the actual Clustering in Orthogonal Spaces (Orth2) procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : list
        list containing number of clusters for each subspace
    explained_variance_for_clustering : float
        Defines the variances that is contained in the subspace used for clustering. If this value is 1, PCA will not be executed before performing KMeans
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, list, list, list, np.ndarray)
        The labels,
        The cluster centers,
        The projections,
        The PCA transformations,
        The mean value of the data set
    """
    assert (
        explained_variance_for_clustering > 0 and explained_variance_for_clustering <= 1
    ), "explained_variancefor_clustering must be within (0, 1)"
    labels = np.zeros((X.shape[0], len(n_clusters)), dtype=np.int32)
    cluster_centers = []
    projections = []
    pcas = [] if explained_variance_for_clustering != 1 else None
    # Center data
    global_mean = np.mean(X, axis=0)
    X = X - global_mean
    for subspace, k in enumerate(n_clusters):
        # (Optional) Execute PCA before clustering
        if explained_variance_for_clustering != 1:
            pca = PCA(explained_variance_for_clustering)
            X_subspace = pca.fit_transform(X)
            pcas.append(pca)
        else:
            X_subspace = X
        # Execute clustering
        km = KMeans(k, random_state=random_state)
        km.fit(X_subspace)
        # Save labels
        labels[:, subspace] = km.labels_
        # Get orthogonal space. Note that due to PCA, KMeans centers can be lower-dimensional
        centers_subspace = np.array(
            [np.mean(X[km.labels_ == c], axis=0) for c in range(k)]
        )
        cluster_centers.append(centers_subspace)
        pca_subspace = PCA()
        pca_subspace.fit(centers_subspace)
        A = pca_subspace.components_[: min(k - 1, X.shape[1])]
        P = np.identity(A.shape[1]) - A.T @ np.linalg.inv(A @ A.T) @ A
        X = X @ P
        projections.append(P)
    return labels, cluster_centers, projections, pcas, global_mean


class OrthogonalClustering(BaseEstimator, ClusterMixin):
    """
    Execute the Orthogonal Clustering procedure (Orth1).
    The algorithm will search for multiple clustering solutions by transforming the feature space after each KMeans execution.
    The number of subspaces will automatically be traced by the length of the input n_clusters array.

    Parameters
    ----------
    n_clusters : list
        list containing number of clusters for each subspace
    explained_variance_for_clustering : float
        Defines the variance that is contained in the subspace used for clustering. This subspace is received by performing PCA.
        If explained_variance_for_clustering is 1, PCA will not be executed before performing KMeans (default: 0.9)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : list
        The final cluster centers
    projections_ : list
        The orthogonal projections
    PCAs_ : list
        The PCA transformations
    global_mean_ : np.ndarray
        The mean value of the fitted data set


    References
    ----------
    Cui, Ying, Xiaoli Z. Fern, and Jennifer G. Dy. "Non-redundant multi-view clustering via orthogonalization."
    Seventh IEEE international conference on data mining (ICDM 2007). IEEE, 2007.
    """

    def __init__(
        self,
        n_clusters: list,
        explained_variance_for_clustering: float = 0.9,
        random_state: np.random.RandomState = None,
    ):
        self.n_clusters = n_clusters
        self.explained_variance_for_clustering = explained_variance_for_clustering
        self.random_state = check_random_state(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "OrthogonalClustering":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : OrthogonalClustering
            this instance of the OrthogonalClustering algorithm
        """
        labels, centers, projections, pcas, global_mean = _orthogonal_clustering(
            X,
            self.n_clusters,
            self.explained_variance_for_clustering,
            self.random_state,
        )
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.projections_ = projections
        self.PCAs_ = pcas
        self.global_mean_ = global_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of an input dataset. For this method the results from the fit() method will be used.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        predicted_labels : np.ndarray
            the predicted labels of the input data set for each subspace. Shape equals (n_samples x n_subspaces)
        """
        # Check if algorithm has run
        assert hasattr(
            self, "labels_"
        ), "The algorithm has not run yet. Use the fit() function first."
        predicted_labels = np.zeros((X.shape[0], len(self.n_clusters)), dtype=np.int32)
        # Get labels for each subspace
        for subspace in range(len(self.n_clusters)):
            X_transform = self.transform_subspace(X, subspace)
            print("pred orth1 -", subspace, [x for x in X_transform])
            if self.PCAs_ is not None:
                X_transform = self.PCAs_[subspace].transform(X_transform)
                centers_subspace = self.PCAs_[subspace].transform(
                    self.cluster_centers_[subspace]
                )
            else:
                centers_subspace = self.cluster_centers_[subspace]
            labels_tmp, _ = pairwise_distances_argmin_min(
                X=X_transform,
                Y=centers_subspace,
                metric="euclidean",
                metric_kwargs={"squared": True},
            )
            predicted_labels[:, subspace] = labels_tmp
        # Return the predicted labels
        return predicted_labels

    def transform_subspace(self, X: np.ndarray, subspace_index: int) -> np.ndarray:
        """
        Transform the input dataset with the projections identified by the fit function.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        subspace_index : int
            the index of the specific subspace

        Returns
        -------
        X : np.ndarray
            The transformed dataset
        """
        assert subspace_index < len(
            self.n_clusters
        ), "subspace_index must be smaller than {0}".format(len(self.n_clusters))
        X = X - self.global_mean_
        for subspace in range(1, subspace_index + 1):
            if self.PCAs_ is not None:
                X_transform = self.PCAs_[subspace - 1].transform(X)
                centers_subspace = self.PCAs_[subspace - 1].transform(
                    self.cluster_centers_[subspace - 1]
                )
            else:
                centers_subspace = self.cluster_centers_[subspace - 1]
                X_transform = X
            labels_tmp, _ = pairwise_distances_argmin_min(
                X=X_transform,
                Y=centers_subspace,
                metric="euclidean",
                metric_kwargs={"squared": True},
            )
            # labels_tmp, _ = pairwise_distances_argmin_min(X=X, Y=self.cluster_centers_[subspace], metric='euclidean',
            #                                               metric_kwargs={'squared': True})
            for c in range(self.n_clusters[subspace - 1]):
                X[labels_tmp == c] = (
                    X[labels_tmp == c] @ self.projections_[subspace - 1][c]
                )
        return X


class ClusteringInOrthogonalSpaces(BaseEstimator, ClusterMixin):
    """
    Execute the Clustering In Orthogonal Spaces procedure (Orth2).
    The algorithm will search for multiple clustering solutions by transforming the feature space after each KMeans execution.
    The number of subspaces will automatically be traced by the length of the input n_clusters array.

    Parameters
    ----------
    n_clusters : list
        list containing number of clusters for each subspace
    explained_variance_for_clustering : float
        Defines the variance that is contained in the subspace used for clustering. This subspace is received by performing PCA.
        If explained_variance_for_clustering is 1, PCA will not be executed before performing KMeans (default: 0.9)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : list
        The final cluster centers
    projections_ : list
        The orthogonal projections
    PCAs_ : list
        The PCA transformations
    global_mean_ : np.ndarray
        The mean value of the fitted data set

    References
    ----------
    Cui, Ying, Xiaoli Z. Fern, and Jennifer G. Dy. "Non-redundant multi-view clustering via orthogonalization."
    Seventh IEEE international conference on data mining (ICDM 2007). IEEE, 2007.
    """

    def __init__(
        self,
        n_clusters: list,
        explained_variance_for_clustering: float = 0.9,
        random_state: np.random.RandomState = None,
    ):
        self.n_clusters = n_clusters
        self.explained_variance_for_clustering = explained_variance_for_clustering
        self.random_state = check_random_state(random_state)

    def fit(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> "ClusteringInOrthogonalSpaces":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : ClusteringInOrthogonalSpaces
            this instance of the ClusteringInOrthogonalSpaces algorithm
        """
        labels, centers, projections, pcas, global_mean = (
            _clustering_in_orthogonal_spaces(
                X,
                self.n_clusters,
                self.explained_variance_for_clustering,
                self.random_state,
            )
        )
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.projections_ = projections
        self.PCAs_ = pcas
        self.global_mean_ = global_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of an input dataset. For this method the results from the fit() method will be used.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        predicted_labels : np.ndarray
            the predicted labels of the input data set for each subspace. Shape equals (n_samples x n_subspaces)
        """
        # Check if algorithm has run
        assert hasattr(
            self, "labels_"
        ), "The algorithm has not run yet. Use the fit() function first."
        predicted_labels = np.zeros((X.shape[0], len(self.n_clusters)), dtype=np.int32)
        # Get labels for each subspace
        for subspace in range(len(self.n_clusters)):
            X_transform = self.transform_subspace(X, subspace)
            if self.PCAs_ is not None:
                X_transform = self.PCAs_[subspace].transform(X_transform)
                centers_subspace = self.PCAs_[subspace].transform(
                    self.cluster_centers_[subspace]
                )
            else:
                centers_subspace = self.cluster_centers_[subspace]
            labels_tmp, _ = pairwise_distances_argmin_min(
                X=X_transform,
                Y=centers_subspace,
                metric="euclidean",
                metric_kwargs={"squared": True},
            )
            predicted_labels[:, subspace] = labels_tmp
        # Return the predicted labels
        return predicted_labels

    def transform_subspace(self, X: np.ndarray, subspace_index: int) -> np.ndarray:
        """
        Transform the input dataset with the projections identified by the fit function.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        subspace_index : int
            the index of the specific subspace

        Returns
        -------
        X : np.ndarray
            The transformed dataset
        """
        assert subspace_index < len(
            self.n_clusters
        ), "subspace_index must be smaller than {0}".format(len(self.n_clusters))
        X = X - self.global_mean_
        for subspace in range(1, subspace_index + 1):
            X = X @ self.projections_[subspace - 1]
        return X
