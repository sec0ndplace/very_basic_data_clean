# EDA and Modeling

The data contained in the marketing_modified.csv file will be used to run an exploratory data analysis and implement machine larning models. A description about the data is found below.

## Context
A response model can provide a significant boost to the efficiency of a marketing campaign by increasing responses or reducing expenses. The objective is to predict who will respond to an offer for a product or service

## Content
  - AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
  - AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
  - AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
  - AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
  - AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
  - Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
  - Complain - 1 if customer complained in the last 2 years
  - DtCustomer - date of customer’s enrolment with the company
  - Education - customer’s level of education
  - Marital - customer’s marital status
  - Kidhome - number of small children in customer’s household
  - Teenhome - number of teenagers in customer’s household
  - Income - customer’s yearly household income
  - MntFishProducts - amount spent on fish products in the last 2 years
  - MntMeatProducts - amount spent on meat products in the last 2 years
  - MntFruits - amount spent on fruits products in the last 2 years
  - MntSweetProducts - amount spent on sweet products in the last 2 years
  - MntWines - amount spent on wine products in the last 2 years
  - MntGoldProds - amount spent on gold products in the last 2 years
  - NumDealsPurchases - number of purchases made with discount
  - NumCatalogPurchases - number of purchases made using catalogue
  - NumStorePurchases - number of purchases made directly in stores
  - NumWebPurchases - number of purchases made through company’s web site
  - NumWebVisitsMonth - number of visits to company’s web site in the last month
  - Recency - number of days since the last purchase

## Acknowledgements
O. Parr-Rud. Business Analytics Using SAS Enterprise Guide and SAS Enterprise Miner. SAS Institute, 2014.

## Data Source
https://www.kaggle.com/rodsaldanha/arketing-campaign

Keep in mind that the data that has been given is slightly modified from that on Kaggle.


to load the data in a jupyter notebook and perform a exploratory data analysis. The analysis should be thorough with commentary on what was done and why. Some of the things you will want to include are
  - What are the datatypes?
  - Is there missing data?
  - How many levels are there in factor levels?
  - Are the factor sparse in some levels?
  - How spread is the numerical data?
  - Are there any outliers?

## Exploratory Data Analysis

The EDA peformed should be in depth with plots, statistics, and commentary on what was done and why. Each dataset is different which nessisitates different things will be done than what we covered in class.

## Modeling

Select at least 2 machine learning models in the sklearn package to predict if the customer accepted an offer in the last marketing campaign. Describe (in your notebook) why you chose the models you did. How do your models perform? Is there other data/information that would be useful in getting a better prediction?