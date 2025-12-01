# Final Report - COVID 19 Misinformation Classification

_Final PIT for Elective 4 - DSS._

Section: 4R8

Group Members:

- Mosqueda, Christine Reisa P.
- Viernes, Jhon Lloyd D.
- Gutang, Bobby John

---

# Rationale/Background

During the peak of COVID 19, numerous news articles and academic studies have pointed out how misinformation spreads fast in the digital space. This _infodemic_ has caused distraught, confusion, and panic towards the masses, thereby necessitating for finding viable solutions to mitigate the spread of misinformation. Intriguingly, the academic and technological landscape have embraced the application of Artificial Intelligence in the intersection of healthcare and media. Such applications lead to the integration of AI in the digital space as one of the viable options for mitigating the spread of misinformation, where AI models have been deployed in the real-world industry to help combat the staggering number of misinformation in the internet. Machine Learning models like Naive Bayes, Support Vector Machines, LibLinear, and LibShortText have been utilized for classification tasks related to COVID-19 Misinformation (Cartwright, 2023).

Altogether, this project positions itself as an exploratory analysis and evaluation about the performance of different types of Machine Learning models in COVID-19 Misinformation Classification.

# Objectives

- To obtain and use existing raw datasets from reputable and credible resources.
- To clean and preprocess the obtained raw datasets.
- To train and test different text classification models (classical and deep learning) on the preprocessed dataset.
- To analyze and evaluate the performance of different models.

# Methodology

## Dataset Collection

For the dataset collection, the group had found reputable and credible dataset resources from United Nations Educational, Scientific and Cultural Organization (UNESCO) and University of California, Irvine (UCI). Both dataset consists of more than 5000 rows, which can already suffice for the training of text classification models. Imbalance in class distribution is also anticipated as in the real-world setting, as some significant classes tend to outnumber others. Both datasets are also labeled in multi-class. However, each has different scope and goal even though their use cases fall under the same genre of COVID-19 Misinformation Classification.

To compensate for the tradeoffs, each dataset were individually and separately trained on the selected models. For example, UNESCO dataset and UCI dataset were both trained on Logistic Regression in separation, which technically accounts to two models but trained on two different datasets.

This approach aims for the model training to be efficient and test the variability of the models' performance when trained and exposed to different datasets with different scopes.

## Model Selection

The group decided to explore two different types of machine learning classification models: (1) classic models, and (2) deep learning models. The performance of classic models (or traditional ML models), served as the baseline for comparing the differences of text classification models.

**Classic Models:**

- Logistic Regression
- Naive Bayes (Multinomial Naive Bayes)
- Random Forest (Ensemble)
- SVM (Linear Kernel)

**Deep Learning Models:**

- Convolutional Neural Network (in Text)
- DistilBERT (Pre-trained Transformer)
- Bidirectional Long Short Term Memory

## Evaluation Metrics

Given the nature of both datasets and the anticipation of class distribution imbalance, the group has identified the **F1 Score** as the most critical metric to consider, next to Accuracy. F1 score provides a more balanced evaluation of a model's performance, especially on imbalanced datasets, by considering both precision and recall. Whereas accuracy can be misleading especially when such class was favored or outnumbers other classes, thereby only predicting the majority. Confusion matrix report were also produced and evaluated to better grasp and understand the bigger picture of the models' performance.

## Training Procedure

1. Explore and analyze the raw dataset.
2. Clean the raw dataset by applying text cleaning and preprocessing techniques on the corpus.
3. Preprocess the corpus.
4. Apply label encoding on the class labels.
5. Feature selection and extraction.
6. Text tokenization.
7. Train-test splitting.
8. Model parameter configuration.
9. Model compilation
10. Model fitting (training).
11. Evaluate model performance using metrics like f1 score, accuracy, and plotting a confusion matrix for visualization.
12. Save evaluation report and other assets like model, vectorizers, and tokenizers.

---

# Dataset Information

## UNESCO - ESOC COVID 19 Misinformation Dataset

- **Link:** [UNESCO - ESOC COVID 19 Misinformation Dataset](https://www.unesco.org/en/world-media-trends/esoc-covid-19-misinformation-dataset?fbclid=IwY2xjawOWKH1leHRuA2FlbQIxMABicmlkETFreWZKcmdmZG5icHNQNXVpc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHpc8H5CCH3ld4rLBQvhawZv_VupMfCPHVFhWh_cYyqxvOB7v7JNmqLafClyy_aem_TXVN4cnCy4egHdiJ0rL48A)
- **About:** COVID 19 misinformation dataset collected on social media and news outlets around the world, from the early days of pandemic up to December 2020.
- **Size:** 5613 rows
- **Classification Type:** Multiclass
- **Type of data:** Qualitative
- **Type of data collection:** Crowdsourcing, experts
- **Class Distribution:**

```
False reporting - 4123
Conspiracy - 966
Fake remedy - 502
Conspiracy, False reporting - 5
False Reporting - 3
false reporting - 3
Fake remedy, false reporting - 2
False reporting, Fake remedy - 1
Conspiracy, Fake remedy - 1
Fake remedy, False reporting - 1
False reporting, Conspiracy - 1
Fake remedy, conspiracy - 1
```

- **Class Imbalance:** Highly skewed, with False reporting class outnumbering other classes. There is also an anomaly in label annotation.

## UCI Covid19-Lies

- **Link:** [UCINLP Covid19-Lies](https://github.com/ucinlp/covid19-data?fbclid=IwY2xjawOWJLpleHRuA2FlbQIxMABicmlkETFreWZKcmdmZG5icHNQNXVpc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHm6w1xZNzH_oMufcfadEBPxyGC-XRBQUoJm8O8NH9yziIIq6Diim5_-4nqlz_aem_aLfdx-XOS7RoDItT-uRKvQ)
- **About:** Misinformation dataset about misconceptions of COVID 19 collected on X (formerly Twitter).
- **Size:** 6591 rows
- **Classification Type:** Multiclass
- **Type of data:** Qualitative
- **Type of data collection:** Annotation by researchers
- **Class Distribution:**

```
na - 6149
pos - 288
neg - 154
```

- **Class Imbalance:** Highly skewed, with na (no stance) class outnumbering other classes. Labeling is clear.

# Other Strategies

- **Splitting:** Train set (80%), Test set (20%)
- **ngram range (for classical):** (1,2)
- **Random seed:** 42
- **Oversampling methods:** None / Random Oversampling / Borderline SMOTE

# Results and Discussion

# Conclusion

---

# References

- Cartwright, B., Frank, R., Weir, G., Padda, K., & Strange, S.-M. (2023). Deploying Artificial Intelligence to Combat Covid-19 Misinformation on Social Media: Technological and Ethical Considerations. Proceedings of the ... Annual Hawaii International Conference on System Sciences/Proceedings of the Annual Hawaii International Conference on System Sciences. https://doi.org/10.24251/hicss.2023.266

‌
