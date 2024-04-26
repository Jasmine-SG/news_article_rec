# Personalized News Article Recommendation System Using GCNs

## Project Description
This repository houses the codebase for our research on a personalized news article recommendation system developed using Graph Convolutional Networks (GCNs). The system is designed to predict user preferences for digital news articles by constructing a heterogeneous graph that integrates user and article nodes, employing features such as textual content, user demographics, browsing history, session details, and engagement metrics. This approach leverages the complex dynamics of user-article interactions and the potential of GCNs in enhancing digital platform recommendation systems.

## Background
Recommender systems are critical in helping users navigate extensive digital content. This project, initiated as part of the Ekstra Bladet RecSys Challenge, addresses the challenges of predicting user clicks on news articles. Our GCN-based system differs from traditional recommender systems by constructing a heterogeneous graph that captures intricate relationships and interactions through rich feature integration. This method allows for the nuanced capture of user preferences and behaviors, significantly enhancing recommendation accuracy.

## System Overview
- **Graph Construction**: Utilizes a comprehensive dataset from the Ekstra Bladet RecSys Challenge, encompassing detailed user demographics, browsing history, and article content, to construct a feature-rich heterogeneous graph.
- **GCN Model**: Applies graph convolutional networks to derive robust embeddings for nodes, effectively capturing the subtle interactions necessary for precise recommendations.
- **Predictive Model**: Employs the embeddings to predict the likelihood of user-article clicks, facilitating personalized content delivery.

## Key Features

### Embedding Techniques
1. **User-Article Interaction Embedding:**
   - Combines features like article IDs in view, read time, and scroll percentage into a single vector per user, encapsulating user engagement.

2. **Article Content Embedding:**
   - Employs BERT embeddings to transform article titles, subtitles, and content into numerical vectors that capture semantic meanings.

3. **Categorical Feature Embedding:**
   - Transforms categorical data such as device types and demographics into numerical representations using one-hot encoding and dimensionality reduction techniques like feature hashing.

4. **Numerical Feature Embedding:**
   - Normalizes engagement metrics (inviews, read times, scroll percentages) using min-max scaling to ensure uniform scale processing.

5. **Temporal Feature Embedding:**
   - Standardizes and normalizes time-based features (e.g., article publication times) to help the model capture temporal patterns.

6. **Context Preservation:**
   - Tracks the relationship between articles viewed and clicked, maintaining contextual integrity to enhance predictive accuracy.

### Implementation Details
- Feature vectors are stored in dictionaries keyed by article and user IDs, used to create padded feature tensors ensuring uniform dimensions. These tensors are concatenated to form the final node features tensor, assigned to the graph as `g.ndata['feat']`.

This structured approach to feature embedding not only facilitates efficient data handling but also ensures that the model can effectively learn from complex user-article interactions. Scripts and detailed documentation for these techniques are available in this repository, aiding in both research and practical application development in news recommendation systems.

### Model Architecture

We utilize a **Graph Convolutional Network (GCN)**, based on the work by Kipf and Welling (2016), as the core of our model architecture. The GCN operates on a graph **G**, processing learned embeddings to generate node embeddings that reflect the graph's structural information. The architecture is designed with flexibility, allowing adjustments in the number of layers and hidden features to tailor the model's capacity and expressiveness.

The implementation uses the **GraphConv** module from **DGL**, which facilitates message passing and feature aggregation among nodes. Each graph convolutional layer within our model applies a linear transformation followed by a non-linear activation function, such as **ReLU**, to the node features. This process updates the node embeddings to encapsulate local neighborhood information effectively.

To assess the probability of a user clicking on an article, we incorporate a **link predictor module**. This module, designed with fully connected layers featuring ReLU activation and dropout regularization (Srivastava et al. 2014), aims to mitigate overfitting. It processes the embeddings from user and article nodes, outputting a probability score that predicts the likelihood of a user-article interaction.

### Future Work
Further enhancements will focus on refining sampling techniques, advancing feature engineering, and exploring more sophisticated GCN architectures. We aim to incorporate richer content analyses and robust cross-validation methods to ensure model reliability and extend its applicability.

By addressing both theoretical and practical aspects, this project contributes to the ongoing development of advanced recommender systems, aiming to significantly improve user engagement and satisfaction in digital news consumption.
