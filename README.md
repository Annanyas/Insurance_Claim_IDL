# **Deep Claim Extensions for Medicare FFS Claims**

This repository contains the implementation and evaluation of enhancements to the Deep Claim model, tailored for predicting claim responses on Medicare Fee-for-Service (FFS) datasets. This project is part of the **IDL (Intelligent Data Lab) coursework**, aimed at advancing state-of-the-art techniques in healthcare claims processing.

---

## **Table of Contents**
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Dataset](#dataset)  
4. [Evaluation Metrics](#evaluation-metrics)  
5. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
6. [Usage](#usage)  
7. [Contributors](#contributors)  
8. [Acknowledgments](#acknowledgments)  
9. [License](#license)  

---

## **Project Overview**

Health insurance claim denial rates and response times are critical for ensuring financial efficiency in healthcare. Building on the baseline Deep Claim model, this project adapts the framework to Medicare FFS claims, addressing dataset-specific challenges and exploring advanced techniques to improve prediction accuracy and interpretability.

---

## **Key Features**

### **Baseline Model**
- Adaptation of the **Deep Claim** architecture:
  - **Claims Embedding Network**: Converts sparse claim data into dense representations.
  - **Multi-Task Learning**: Predicts claim denial probability, denial reasons, and response time.
  - **Interpretability**: Highlights key claim features influencing denial outcomes.

### **Proposed Enhancements**
1. **Integration of LLMs**: 
   - Parse free-text fields in claims and generate actionable suggestions using models like GPT-4.
2. **Incorporating Historical Claims**:
   - Aggregate multiple claims from the same patient to capture patterns and dependencies.
3. **Medicare-Specific Embeddings**:
   - Tailor embeddings for Medicare FFS codes and claim structures.
4. **Dynamic Multi-Task Learning**:
   - Dynamically adjust task weights and introduce auxiliary predictions, such as recovery rates for denied claims.

---

## **Dataset**

The project uses **Medicare FFS claims data**. Key challenges include:
- Variability in claim formats and denial reasons.
- Integrating historical claim data for patient-level context.

---

## **Evaluation Metrics**

The following metrics are used to assess model performance:
- **Precision-Recall Area Under Curve (PR-AUC)**: Handles imbalanced datasets effectively.
- **Recall at 95% Precision**: Focuses on actionable predictions with minimal false positives.
- **Mean Absolute Error (MAE)**: Evaluates response time prediction accuracy.





## **Contributors**
- **Annanya** 
- **Aimee Langevin**  
- **Weiqian Zhang**   

---

## **Acknowledgments**
This project is conducted as part of the **IDL course** at CMU. The baseline model is adapted from the original Deep Claim paper.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
