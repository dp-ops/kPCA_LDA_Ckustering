import torch

class NearestCentroid():
    def __init__(self): 
        pass
        

    def _check_classification_targets(self, y):
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype = torch.int64)
        
        if len(y.shape) != 1:
            raise ValueError(f"Target labels must be one-dimensional, but got shape {y.shape}")
        
        unique_labels = torch.unique(y)
        if unique_labels.dtype != torch.int:
            raise ValueError("Target labels must be integers.")
        
        if len(unique_labels) < 2:
            raise ValueError("The number of classes must be greater than one.")
        
        return y
                             
    def fit(self, x: torch.tensor, y: torch.tensor):
        self._check_classification_targets(y)
        
        x = x.clone().detach().to(torch.float32)

        n_samples, n_features = x.shape
        unique_classes = torch.unique(y)
        self.classes = unique_classes
        n_classes = len(unique_classes)
        
        #mask mapping each class to its member
        self.centroids = torch.empty((n_classes, n_features), dtype=torch.float32)
        nk = torch.zeros(n_classes, dtype=torch.float32)

        for i, c_classes in enumerate(unique_classes):
            center_mask = (y == c_classes)
            nk[i] = center_mask.sum()
            #euclidean metric
            self.centroids[i] = x[center_mask].mean(dim=0)

        return self
    
    def predict(self, x: torch.tensor):
        x = x.clone().detach().to(torch.float32)
        dist = torch.cdist(x, self.centroids, p=2)
        return self.classes[torch.argmin(dist, dim=1)]
    

class Kmeans():
    
    def __init__(self, n_clusters = 8, tolerance = 0.0001, max_iter = 100, runs = 1, init_method = 'forgy'):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = torch.zeros(n_clusters)
        self.max_iter = max_iter

    #mporei na thelei na to afereseis
        self.runs = runs if init_method == 'forgy' else 1


    # def _initialize_means(self, x:torch.tensor, row_count):
    def forgy(self, x:torch.tensor, row_count, n_clusters):
        return x[torch.randperm(x.size(0))[:n_clusters]]
    
    def _compute_dist(self, x:torch.tensor, cluster_means, row_count):
        dist = torch.zeros((row_count, self.n_clusters))
        x1 = x.to(torch.float32)
        for cluster_mean_i, cluster_mean in enumerate(cluster_means):
            dist[:, cluster_mean_i] = torch.norm(x1 - cluster_mean, dim=1)
        
        return dist
    
    def _label_examples(self, dist):
        return dist.argmin(dim=1)
    
    def _compute_means(self, x:torch.tensor, labels, col_count):
        cluster_means = torch.zeros((self.n_clusters,col_count))
        x1 = x.to(torch.float32)
        for cluster_mean_i, _ in enumerate(cluster_means):
            cluster_elements = x1[labels == cluster_mean_i]
            if len(cluster_elements):
                cluster_means[cluster_mean_i, :] = cluster_elements.mean(dim = 0)
        
        return cluster_means
    
    def _compute_cost(self, x:torch.tensor, labels, cluster_means):
        cost = 0
        for cluster_mean_i, cluster_mean in enumerate(cluster_means):
            cluster_elements = x[labels == cluster_mean_i]
            cost += torch.norm(cluster_elements - cluster_mean, dim=1).sum()

        return cost 
    
    def _get_values(self, x:torch.tensor):
        # if isinstance(x, torch.tensor):
        #     return x
        return torch.tensor(x)
    
    def fit(self, x:torch.tensor, y:torch.tensor):
        row_count, col_count = x.shape
        X_values = self._get_values(x)
        x_labels = torch.zeros(row_count)

        costs = torch.zeros(self.runs)
        all_clusterings = []

        for i in range(self.runs):
            cluster_means = self.forgy(X_values, row_count, n_clusters=self.n_clusters)

            for _ in range(self.max_iter):
                previous_means = cluster_means.clone()
                dist = self._compute_dist(X_values, cluster_means, row_count)
                
                x_labels = self._label_examples(dist)
                cluster_means = self._compute_means(X_values, x_labels, col_count)
                clusters_not_changed = torch.abs(cluster_means - previous_means) < self.tolerance

                if torch.all(clusters_not_changed) != False:
                    break

            X_labels = X_labels.unsqueeze(1)  # Add a new axis to X_labels (equivalent to np.newaxis)
            X_values_with_labels = torch.cat((X_values, X_labels), dim=1)  # Concatenate along columns (dim=1)

            all_clusterings.append((cluster_means, X_values_with_labels))
            costs[i] = self._compute_cost(X_values, x_labels, cluster_means)

        best_clustering_i = costs.argmin()

        self.cost = costs[best_clustering_i]

        return all_clusterings[best_clustering_i]