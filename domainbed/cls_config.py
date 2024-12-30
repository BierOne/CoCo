"""
Create Time: 7/8/2022
Author: BierOne (lyibing112@gmail.com)
"""


clustering_param = {
    "PACS": {
        "cluster_steps": [1000, 2000, 3000, 4000, 5002],
        "kmeans": {
            "label_ids": range(7),
            "reg_layer_list": ["layer4"],
            "clustering_method": 'kmeans',
            "low_level_merge_method": "jaccard",
            "high_level_merge_method": "jaccard",
            "domain_level": False,
            "class_level": True,
            "multi_level": True,
            "random": False,
            "distance_metric": "euclidean",
            "num_concept_clusters": 10,
            "max_cluster_size": 100,
            "quantile": 0.1,
            "act_ratio": 0.3
        },
    },
    "VLCS": {
        "cluster_steps": [1000, 2000, 3000, 4000, 5002],
        "kmeans": {
            "label_ids": range(5),
            "reg_layer_list": ["layer4"],
            "clustering_method": 'kmeans',
            "low_level_merge_method": "jaccard",
            "high_level_merge_method": "jaccard",
            "domain_level": False,
            "class_level": True,
            "multi_level": True,
            "random": False,
            "distance_metric": "euclidean",
            "num_concept_clusters": 5,
            "max_cluster_size": 50,
            "quantile": 0.1,
            "act_ratio": 0.05
        }
    },
    "OfficeHome": {
        "cluster_steps": [1000, 2000, 3000, 4000, 5002],
        "kmeans": {
            "label_ids": range(65),
            "reg_layer_list": ["layer4"],
            "clustering_method": 'kmeans',
            "low_level_merge_method": "jaccard",
            "high_level_merge_method": "dup_filtering",
            "domain_level": False,
            "class_level": True,
            "multi_level": True,
            "random": False,
            "distance_metric": "euclidean",
            "num_concept_clusters": 5,
            "max_cluster_size": 50,
            "quantile": 0.01,
            "act_ratio": 0.3
        }
    },
    "DomainNet": {
        "cluster_steps": [3000, 4000, 5002],
        "kmeans": {
            "label_ids": range(345),
            "reg_layer_list": ["layer4"],
            "clustering_method": 'kmeans',
            "low_level_merge_method": "jaccard",
            "high_level_merge_method": "jaccard",
            "domain_level": False,
            "class_level": True,
            "multi_level": False,
            "random": False,
            "distance_metric": "euclidean",
            "num_concept_clusters": 1,
            "max_cluster_size": 30,
            "quantile": 0.1,
            "act_ratio": 0.1
        }
    },

    "TerraIncognita": {
        "cluster_steps": [1000, 2000, 3000, 4000, 5002],
        "kmeans": {
            "label_ids": range(10),
            "reg_layer_list": ["layer4"],
            "clustering_method": 'kmeans',
            "low_level_merge_method": "jaccard",
            "high_level_merge_method": "jaccard",
            "domain_level": False,
            "class_level": True,
            "multi_level": True,
            "random": False,
            "distance_metric": "euclidean",
            "num_concept_clusters": 10,
            "max_cluster_size": 20,
            "quantile": 0.05,
            "act_ratio": 0.3
        }
    },
}

