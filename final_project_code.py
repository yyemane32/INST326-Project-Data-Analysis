#For Data Analysis
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#For Data Visulaization
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
df = pd.read_csv('GSS.csv')


#Dropping Unwanted columns
df.drop(columns= ["R's family's negative attitudes about mh problems", 
                  "Ballot used for interview", "Respondent id number", "Respondents income",'Total family income'], inplace = True)
df.drop(df.columns[0], axis=1, inplace=True)

#Dropping NA values
df.dropna(inplace = True)

#Renaming columns
df.rename(columns={'Rs income in constant $': 'Income', 
                       'R is how tall': 'Height','R weighs how much': 'Weight','Rs religious preference': 'Religion',
                        'Number of persons in household': 'Number of Household',
                         'Race of respondent': 'Race', 'Respondents sex': 'Sex', 'Rs highest degree': 'Degree Completed',
                        'Age of respondent': 'Age', 'Number of children': 'Children'}, inplace = True)


#Replacing values in dataframe
df['Degree Completed'].replace({"Lt high school": 'High school',  "Junior college": 'Under Grad',"Bachelor": 'Under Grad'}, inplace=True)
df['Marital status'].replace({'Never married':'Unmarried',  'Separated':'Unmarried', 'Divorced':'Unmarried', 'Widowed':'Unmarried'}, inplace = True)
df['Labor force status'].replace({'Temp not working':'Unemployed',  'Keeping house':'Unemployed', 'School':'Unemployed', 'Other':'Unemployed',  'Unempl, laid off': 'Unemployed'}, inplace = True)
df['Religion'].replace({'Hinduism':'Other', 'Inter-nondenominational':'Other', 'Moslem/islam':'Other', 'Native american': "Other",'Orthodox-christian': 'Other', 'Other': "Other", 'Other eastern': "Other" }, inplace = True)

#changing categorical variables to Dummy variables
new_df = pd.get_dummies(df, columns=["Religion","Race","Sex","Degree Completed", 'Marital status', 'Labor force status'])

#Dropping unwanted dummy variables
new_df.drop(columns= ["Labor force status_No answer", "Marital status_No answer","Religion_Don't know", "Religion_No answer", "Degree Completed_Don't know", "Degree Completed_No answer"], inplace = True)


#Indexing and dropping more unwanted rows

new_df = new_df[new_df.Children != 'Eight or more']
new_df = new_df[new_df.Children != 'Dk na']
new_df = new_df[new_df.Age != '89 or older']
new_df = new_df[new_df.Age != 'No answer']
weight = new_df[new_df['Weight'] == 'Not applicable' ].index
weight_no = new_df[new_df['Weight'] == 'No answer' ].index
weight_not = new_df[new_df['Weight'] == "Don't know" ].index

new_df.drop(weight,  inplace=True)
new_df.drop(weight_no,  inplace=True)
new_df.drop(weight_not,  inplace=True)

height = new_df[new_df['Height'] == 'Not applicable' ].index
height_no = new_df[new_df['Height'] == 'No answer' ].index
height_not = new_df[new_df['Height'] == "Don't know" ].index


new_df.drop(height,inplace=True)
new_df.drop(height_not,inplace=True)
new_df.drop(height_no,inplace=True)

#Changing the datatypes of the columns
new_df[['Height', 'Weight', 'Age', 'Children']] = new_df[['Height', 'Weight', 'Age', 'Children']].apply(pd.to_numeric) 


#Creating a Linear regression function
def regression(independent, dependent):
    X =new_df.loc[:, independent].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = new_df.loc[:, dependent ].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    
    
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    

    plt.scatter(X, Y)
    plt.title("Linear regression of " + dependent + ' against ' + independent)
    plt.xlabel(independent)
    plt.ylabel(dependent)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    
    r = model.rsquared * 100
    rounded = r.round(2)
    print(str(rounded) + "% of the variation in " + dependent + " is explained by " + independent)

    X_prediction = float(input ("What is your " + independent + " ? "))
    equation = (int(linear_regressor.coef_ [0][0])* X_prediction) + int(linear_regressor.intercept_[0]) 
    print("Your predicted" ,dependent ,"is", equation)
    
    
#Calling out the function
#independent_input = input('Please input independent variable: ')
#dependent_input = input('Please input dependent variable: ')
#regression(independent_input, dependent_input)

#Logistic model for only one categrical or numerical variable
def logistic(dependent):
    Y = df.loc[:, dependent].values.reshape(-1, 1)
    plt.figure(figsize=(15, 10))
    sns.countplot(x= dependent, data=df)
    plt.show()

#dependent = input(" Insert a categorical variable: ")
#logistic(dependent)
    
#Logistical model for numerical vs categorical variable
def logistic1(independent, dependent):
    df[['Height', 'Weight', 'Age', 'Children']] = new_df[['Height', 'Weight', 'Age', 'Children']].apply(pd.to_numeric) 
    X =df.loc[:, independent].values.reshape(-1, 1)  
    Y =df.loc[:, dependent ].values.reshape(-1, 1)
    plt.figure(figsize=(15, 10))
    sns.boxplot(x= independent,y= dependent,data= df,palette='winter')
    plt.show()
    
#independent = input(" Put your independent variable that is numerical:")
#dependent = input(" Put your dependent variable that is categorical: ")
#print(logistic1(independent, dependent))


if __name__ == "__main__":
    print("Below are the column lists from the dataframe")
    for col in new_df.columns:
        print(col)
    independent_input = input('Please choose an independent variable from above :')
    dependent_input = input('Please choose a dependent variable from above : ')
    regression(independent_input, dependent_input)
    
    print("Below are the column lists from the dataframe")
    for cols in df.columns:
        print(cols)
    dependent = input(" Insert a categorical variable: ")
    logistic(dependent)
    
    independent = input(" Put your independent variable that is numerical:")
    dependent = input(" Put your dependent variable that is categorical: ")
    print(logistic1(independent, dependent))
    
    