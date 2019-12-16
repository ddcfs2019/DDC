### Introduction
In many applications of business and marketing analytics, predictive models are fit
using hierarchically structured data: common characteristics of products, customers, or
webpages are represented as categorical variables, and each category can be split up into
multiple subcategories at a lower level of the hierarchy. The model may thus contain
hundreds of thousands of binary variables, necessitating the use of variable selection
to screen out large numbers of irrelevant or insignificant features. We propose a new
dynamic screening method, based on the distance correlation criterion, designed for
hierarchical binary data. Our method can screen out large parts of the hierarchy at
the higher levels, avoiding the need to explore many lower-level features and greatly
reducing the computational cost of screening. The practical potential of the method is
demonstrated in a case application on user-brand interaction data from Facebook.


- Various feature selection algorithms, including Lasso, Sure Independence Screening (SIS), Streamwise (SR), False Discovery Rate (FDR), and our proposed DDC.

- We use both simulation and real data from Facebook to validate our proposed algorithm DDC, with comparison to all above baselines regarding effectiveness and efficiency.

### Content

- data:
  - data/simulation
  - data/real
  
- codes:
  - real/: all implemented algorithms on the real data (small-scale and large-scale)
   
  - simulation/: all implemented algorithms on the simuated data (100 datasets for two different scales: n=100, p=5,500 and n=1,000, p=170,000)
  

### Prerequisite
- Python 3.x with the following packages installed
  - Numpy
  - Scikit-learn
