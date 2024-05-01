
# Search Functionality for Gita Learn

This document provides a comprehensive overview of the implementation of a Bhagavad Gita search functionality using a machine learning model. The system allows users to input a query, which is then matched against verses from the Bhagavad Gita text. The matching verses are ranked based on their similarity to the query, and the top results are returned to the user.




## Documentation

The purpose of this project is to provide users with an efficient tool to search through the Bhagavad Gita text. By utilizing machine learning techniques, the system aims to deliver accurate and relevant results to users' queries.

## Dependencies

```bash
•	Flask: A micro web framework used for building the application and handling HTTP requests.
•	Flask-CORS: An extension for Flask to handle Cross-Origin Resource Sharing (CORS) headers.
•	scikit-learn: A machine learning library used for vectorization and cosine similarity calculations.
•	NLTK: A natural language processing library used for text preprocessing.
•	Joblib: A library used for saving and loading machine learning models.
•	JSON: A built-in Python library for working with JSON data.

```




## Code Overview

•	4.1 - Preprocessing:
   The Bhagavad Gita verses are preprocessed by tokenizing, stemming, and removing stopwords from the text.

•	4.2 - Vectorization:
    The preprocessed text is then vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) method to represent each verse numerically.

•	4.3 - Cosine Similarity Calculation:
    Cosine similarity is calculated between the user query vector and the TF-IDF matrix containing vectorized Bhagavad Gita verses.

•	4.4 - Search Endpoint:
    The Flask application provides an endpoint /search to handle POST requests containing user queries.
    Upon receiving a query, the system preprocesses it, calculates cosine similarity, and returns the top 5 most relevant verses along with their metadata.


## Deployment

A Flask app was developed and this Flask app is deployed on OnRender, a platform that simplifies the deployment and hosting process for web applications.

#### Deployment Process
```bash
Sign up for an account on OnRender if you haven't already.
Log in to your OnRender account and create a new project.
Configure your project settings, including the name and deployment options.
Connect your GitHub repository to OnRender to enable automatic deployments.
Once connected, OnRender will automatically deploy your Flask app whenever changes are pushed to the connected GitHub repository.

```

#### Monitoring and Logging
OnRender provides built-in monitoring and logging features to help you track the performance of your deployed app. You can access logs and monitor metrics directly from the OnRender dashboard.

#### Scaling and Maintenance

OnRender offers automatic scaling options to handle increased traffic to your app. Maintenance tasks such as updates and patches are managed by OnRender, ensuring your app remains up-to-date and secure.






