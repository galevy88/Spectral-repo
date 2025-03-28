import torch
import numpy as np

from ._utils import *
from sklearn.cluster import KMeans
from ._trainers import SpectralTrainer, SiameseTrainer, AETrainer


class SpectralNet:
    def __init__(
        self,
        n_clusters: int,
        should_use_ae: bool = False,
        should_use_siamese: bool = False,
        is_sparse_graph: bool = False,
        # ae_hiddens: list = [512, 512, 2048, 10],
        ae_hiddens: list = [512, 512, 2048, 26],
        ae_epochs: int = 40,
        ae_lr: float = 1e-3,
        ae_lr_decay: float = 0.1,
        ae_min_lr: float = 1e-7,
        ae_patience: int = 10,
        ae_batch_size: int = 256,
        # siamese_hiddens: list = [1024, 1024, 512, 10],
        siamese_hiddens: list = [1024, 1024, 512, 26],
        siamese_epochs: int = 30,
        siamese_lr: float = 1e-3,
        siamese_lr_decay: float = 0.1,
        siamese_min_lr: float = 1e-7,
        siamese_patience: int = 10,
        siamese_n_nbg: int = 2,
        siamese_use_approx: bool = False,
        siamese_batch_size: int = 128,
        # spectral_hiddens: list = [1024, 1024, 512, 10],
        spectral_hiddens: list = [1024, 1024, 512, 26],
        spectral_epochs: int = 30,
        spectral_lr: float = 1e-3,
        spectral_lr_decay: float = 0.1,
        spectral_min_lr: float = 1e-8,
        spectral_patience: int = 10,
        spectral_batch_size: int = 1024,
        spectral_n_nbg: int = 30,
        spectral_scale_k: int = 15,
        spectral_is_local_scale: bool = True,
    ):
        """SpectralNet is a class for implementing a Deep learning model that performs spectral clustering.
        This model optionally utilizes Autoencoders (AE) and Siamese networks for training.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to be generated by the SpectralNet algorithm.
            Also used for the dimention of the projection subspace.

        should_use_ae : bool, optional (default=False)
            Specifies whether to use the Autoencoder (AE) network as part of the training process.

        should_use_siamese : bool, optional (default=False)
                Specifies whether to use the Siamese network as part of the training process.

        is_sparse_graph : bool, optional (default=False)
            Specifies whether the graph Laplacian created from the data is sparse.

        ae_hiddens : list, optional (default=[512, 512, 2048, 10])
            The number of hidden units in each layer of the Autoencoder network.

        ae_epochs : int, optional (default=30)
            The number of epochs to train the Autoencoder network.

        ae_lr : float, optional (default=1e-3)
            The learning rate for the Autoencoder network.

        ae_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Autoencoder network.

        ae_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Autoencoder network.

        ae_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Autoencoder network.

        ae_batch_size : int, optional (default=256)
            The batch size used during training of the Autoencoder network.

        siamese_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Siamese network.

        siamese_epochs : int, optional (default=30)
            The number of epochs to train the Siamese network.

        siamese_lr : float, optional (default=1e-3)
            The learning rate for the Siamese network.

        siamese_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Siamese network.

        siamese_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Siamese network.

        siamese_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Siamese network.

        siamese_n_nbg : int, optional (default=2)
            The number of nearest neighbors to consider as 'positive' pairs by the Siamese network.

        siamese_use_approx : bool, optional (default=False)
            Specifies whether to use Annoy instead of KNN for computing nearest neighbors,
            particularly useful for large datasets.

        siamese_batch_size : int, optional (default=256)
            The batch size used during training of the Siamese network.

        spectral_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Spectral network.

        spectral_epochs : int, optional (default=30)
            The number of epochs to train the Spectral network.

        spectral_lr : float, optional (default=1e-3)
            The learning rate for the Spectral network.

        spectral_lr_decay : float, optional (default=0.1)
            The learning rate decay factor"""

        self.n_clusters = n_clusters
        self.should_use_ae = should_use_ae
        self.should_use_siamese = should_use_siamese
        self.is_sparse_graph = is_sparse_graph
        self.ae_hiddens = ae_hiddens
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_lr_decay = ae_lr_decay
        self.ae_min_lr = ae_min_lr
        self.ae_patience = ae_patience
        self.ae_batch_size = ae_batch_size
        self.siamese_hiddens = siamese_hiddens
        self.siamese_epochs = siamese_epochs
        self.siamese_lr = siamese_lr
        self.siamese_lr_decay = siamese_lr_decay
        self.siamese_min_lr = siamese_min_lr
        self.siamese_patience = siamese_patience
        self.siamese_n_nbg = siamese_n_nbg
        self.siamese_use_approx = siamese_use_approx
        self.siamese_batch_size = siamese_batch_size
        self.spectral_hiddens = spectral_hiddens
        self.spectral_epochs = spectral_epochs
        self.spectral_lr = spectral_lr
        self.spectral_lr_decay = spectral_lr_decay
        self.spectral_min_lr = spectral_min_lr
        self.spectral_patience = spectral_patience
        self.spectral_n_nbg = spectral_n_nbg
        self.spectral_scale_k = spectral_scale_k
        self.spectral_is_local_scale = spectral_is_local_scale
        self.spectral_batch_size = spectral_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._validate_spectral_hiddens()

    def _validate_spectral_hiddens(self):
        """Validates the number of hidden units in each layer of the Spectral network."""
        print(self.n_clusters)
        print(self.spectral_hiddens[-1])
        if self.spectral_hiddens[-1] != self.n_clusters:
            raise ValueError(
                "The number of units in the last layer of spectral_hiddens network must be equal to the number of clusters or components."
            )

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Performs the main training loop for the SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            Data to train the networks on.

        y : torch.Tensor, optional
            Labels in case there are any. Defaults to None.
        """
        self._X = X
        ae_config = {
            "hiddens": self.ae_hiddens,
            "epochs": self.ae_epochs,
            "lr": self.ae_lr,
            "lr_decay": self.ae_lr_decay,
            "min_lr": self.ae_min_lr,
            "patience": self.ae_patience,
            "batch_size": self.ae_batch_size,
        }

        siamese_config = {
            "hiddens": self.siamese_hiddens,
            "epochs": self.siamese_epochs,
            "lr": self.siamese_lr,
            "lr_decay": self.siamese_lr_decay,
            "min_lr": self.siamese_min_lr,
            "patience": self.siamese_patience,
            "n_nbg": self.siamese_n_nbg,
            "use_approx": self.siamese_use_approx,
            "batch_size": self.siamese_batch_size,
        }

        spectral_config = {
            "hiddens": self.spectral_hiddens,
            "epochs": self.spectral_epochs,
            "lr": self.spectral_lr,
            "lr_decay": self.spectral_lr_decay,
            "min_lr": self.spectral_min_lr,
            "patience": self.spectral_patience,
            "n_nbg": self.spectral_n_nbg,
            "scale_k": self.spectral_scale_k,
            "is_local_scale": self.spectral_is_local_scale,
            "batch_size": self.spectral_batch_size,
        }

        if self.should_use_ae:
            self.ae_trainer = AETrainer(config=ae_config, device=self.device)
            self.ae_net = self.ae_trainer.train(X)
            X = self.ae_trainer.embed(X)

        if self.should_use_siamese:
            self.siamese_trainer = SiameseTrainer(
                config=siamese_config, device=self.device
            )
            self.siamese_net = self.siamese_trainer.train(X)
        else:
            self.siamese_net = None

        is_sparse = self.is_sparse_graph
        if is_sparse:
            build_ann(X)

        self.spectral_trainer = SpectralTrainer(
            config=spectral_config, device=self.device, is_sparse=is_sparse
        )
        self.spec_net = self.spectral_trainer.train(X, y, self.siamese_net)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predicts the cluster assignments for the given data.

        Parameters
        ----------
        X : torch.Tensor
            Data to be clustered.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """
        X = X.view(X.size(0), -1)
        X = X.to(self.device)

        with torch.no_grad():
            if self.should_use_ae:
                X = self.ae_net.encode(X)
            self.embeddings_ = self.spec_net(X, should_update_orth_weights=False)
            self.embeddings_ = self.embeddings_.detach().cpu().numpy()

        cluster_assignments = self._get_clusters_by_kmeans(self.embeddings_)
        return cluster_assignments

    def get_random_batch(self, batch_size: int = 1024) -> tuple:
        """Get a batch of the input data.

        Parameters
        ----------
        batch_size : int
            The size of the batch to use.

        Returns
        -------
        tuple
            The raw batch and the encoded batch.

        """
        permuted_indices = torch.randperm(batch_size)
        X_raw = self._X.view(self._X.size(0), -1)
        X_encoded = X_raw

        if self.should_use_ae:
            X_encoded = self.ae_trainer.embed(self._X)

        if self.should_use_siamese:
            X_encoded = self.siamese_net.forward_once(X_encoded)

        X_encoded = X_encoded[permuted_indices]
        X_raw = X_raw[permuted_indices]
        X_encoded = X_encoded.to(self.device)
        return X_raw, X_encoded

    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Performs k-means clustering on the spectral-embedding space.

        Parameters
        ----------
        embeddings : np.ndarray
            The spectral-embedding space.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments
