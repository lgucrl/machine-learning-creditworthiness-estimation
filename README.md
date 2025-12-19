# Machine Learning estimation of creditworthiness

This repository contains an end-to-end machine learning project to estimate a bank customer’s **creditworthiness** and support decisions on **credit card issuance**. The goal is to build a  model that predicts a binary TARGET where 1 = high creditworthiness (consistent installment payments) and 0 = otherwise, using anonymized customer profile data. The project addresses practical ML challenges, such as handling of mixed data types, class-imbalance strategies and evaluation with metrics that reflect real-world trade-offs.

---

## Dataset

The project is based on a dataset containing **~338,000 records**, where each row represents a customer profile, with the following **19 variables** (mixed numeric + categorical):

- `ID`: customer identification number  
- `CODE_GENDER`: customer's gender  
- `FLAG_OWN_CAR`: indicator of car ownership  
- `FLAG_OWN_REALTY`: indicator of home ownership  
- `CNT_CHILDREN`: number of children  
- `AMT_INCOME_TOTAL`: annual income  
- `NAME_INCOME_TYPE`: type of income  
- `NAME_EDUCATION_TYPE`: level of education  
- `NAME_FAMILY_STATUS`: family status  
- `NAME_HOUSING_TYPE`: type of housing  
- `DAYS_BIRTH`: number of days since birth (commonly stored as a negative value)  
- `DAYS_EMPLOYED`: number of days since the date of hiring (if positive, indicates days since becoming unemployed)  
- `FLAG_MOBIL`: presence of a cell phone number  
- `FLAG_WORK_PHONE`: presence of a work phone number  
- `FLAG_PHONE`: presence of a phone number  
- `FLAG_EMAIL`: presence of an email address  
- `OCCUPATION_TYPE`: type of employment  
- `CNT_FAM_MEMBERS`: number of family members  
- `TARGET`: binary outcome label described above  

The `TARGET` variable is strongly imbalanced (≈91% class `0` vs ≈9% class `1`), so the workflow includes resampling strategies and the selection of metrics suitable for imbalanced classification.

---

## Project workflow
