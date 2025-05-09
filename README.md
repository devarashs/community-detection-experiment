# Community Detection Experiment

## Overview

This repository presents an experimental implementation of community detection algorithms tailored for hypergraphs. It explores a novel approach by transforming hyperedges into a two-section graph using node embedding techniques (Node2Vec and DeepWalk) followed by initial partitioning with K-Means. The resulting graph is then analyzed using three prominent community detection algorithms: Louvain, Leiden, and a hierarchical Louvain variant (H-Louvain). Additionally, the repository includes a pure implementation of the H-Louvain algorithm for direct application on hypergraph data.

This project was initially conceived as a phase one experiment to investigate [briefly mention the broader goal if you're comfortable, e.g., "scalable community detection in complex relational data"]. The current state offers a valuable resource for researchers and practitioners interested in exploring advanced community detection techniques for hypergraphs.

## Key Features

* **Hypergraph to Two-Section Graph Conversion:** Scripts to generate a two-section graph representation from hyperedge data using Node2Vec and DeepWalk embeddings.

* **Initial Partitioning via K-Means:** Implementation of K-Means clustering to obtain initial community assignments for the two-section graph (assumes k=10, easily adjustable).

* **Community Detection Algorithms:**

    * Implementation of the Louvain algorithm.
    * Implementation of the Leiden algorithm.
    * Implementation of the H-Louvain algorithm applied to the generated two-section graph.
    * Pure implementation of the H-Louvain algorithm for direct hypergraph analysis (based on the [pawelwm/h-louvain](https://github.com/pawelwm/h-louvain) repository).

* **Evaluation Metrics:** Automatic calculation and recording of Modularity and Adjusted Mutual Information (AMI) for evaluating the detected communities.

* **Dataset Integration:** Utilizes datasets provided in the [pawelwm/h-louvain](https://github.com/pawelwm/h-louvain) repository for experimentation.

## Potential Future Directions (Optional)

* Experiment with different node embedding techniques and parameters.
* Explore alternative initial partitioning methods.
* Implement additional hypergraph community detection algorithms.
* Develop visualization tools for hypergraphs and their community structures.
* Evaluate performance on a wider range of hypergraph datasets.

## Contributing

Contributions are welcome!  Please feel free to submit pull requests or contact the maintainers directly to discuss potential contributions.

## License

This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

* This work utilizes datasets and the pure H-Louvain implementation from the [H-Louvain](https://github.com/pawelwm/h-louvain.git) repository. The H-Louvain algorithm and hypergraph modularity are described in: [Kamiński et al., 2024](https://doi.org/10.1093/comnet/cnae041).

* The project may also incorporate concepts or algorithms from the [ECCD](https://github.com/bartoszpankratz/ECCD.git) repository. The EC-Louvain/Leiden algorithm is described in: [Pankratz et al., 2024](https://doi.org/10.1093/comnet/cnae035).
