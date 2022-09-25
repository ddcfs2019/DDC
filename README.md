### Introduction

In many applications of business and marketing analytics, predictive models are fit using hierarchically structured data: common characteristics of products, customers, or webpages are represented as categorical variables, and each category can be split up into multiple subcategories at a lower level of the hierarchy. The model may thus contain hundreds of thousands of binary variables, necessitating the use of variable selection to screen out large numbers of irrelevant or insignificant features. We propose a new dynamic screening method, based on the distance correlation criterion, designed for hierarchical binary data. Our method can screen out large parts of the hierarchy at the higher levels, avoiding the need to explore many lower-level features and greatly reducing the computational cost of screening. The practical potential of the method is demonstrated in a case application on user-brand interaction data from Facebook.


- Various feature selection algorithms, including Lasso, Sure Independence Screening (SIS), Streamwise (SR), False Discovery Rate (FDR), and our proposed DDC.

- We use both simulation and real data from Facebook to validate our proposed algorithm DDC, with comparison to all above baselines regarding effectiveness and efficiency.

### Content

- data:
  - data/simulation/data_S_100_5500/     (10 sample files)
  - data/simulation/data_S_1000_170000/ (files are too large to upload. Please email us to get the files or follow our codes to generate)
  - data/real (please follow our code to generate the data under specicif format required by our feature selection procedures)
  
- codes:
  - real/: all implemented algorithms on the real data (small-scale and large-scale)
   
  - simulation/: all implemented algorithms on the simuated data (100 datasets each for two different scales: n=100, p=5,500 and n=1,000, p=170,000)

### Suggested hyperparameter setting

- lambda in Lasso: 0 - 0.1
- d in SIS: 1 - 50 (i.e., [d/log(n)] top percent of features selected)
- q in FDR: 0 - 0.1
- SR-w0 fixed: 0.5, alpha_delta: 0 - 1
- SR-alpha_delta fixed: 0.5, w0: 0 - 1
- Kn in DDC: 0 - 1
  

### Prerequisite
- Python 3.x with the following packages installed
  - Numpy
  - Scikit-learn
