import streamlit as st
import pickle
import pandas as pd
from zipfile import ZipFile
import os
from src import transform_resp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
st.set_page_config(page_title='Credit Score App', page_icon='💰', layout='wide',
                   initial_sidebar_state='auto', menu_items={
                        'Get Help': None,
                        'Report a bug': 'https://github.com/devmedeiros/credit-score-classification-app/issues',
                        'About': '''
                        This app was made by **Jaqueline Medeiros** and its purpose is to showcase how a Credit Score Evaluation work in the fictional bank _Bankio_. 
                        
                        This evaluation is using a Machine Learning model, and you can learn more about how the model work and how I got here by going to the GitHub [repository](https://github.com/devmedeiros/credit-score-classification-app).

                        If you are interested in Data Science you can see follow my work through my blog [devmedeiros.com](https://devmedeiros.com) or on LinkedIn [medeiros-jaqueline](https://www.linkedin.com/in/medeiros-jaqueline/).
                        '''
     })

path = os.path.dirname(__file__)
folder_path = os.path.join(path,'../models')

@st.cache_resource
def load_pickle(file_name):
    try:
        file_path = os.path.join(folder_path, file_name)
        
        #kiểm tra xem file có tồn tại không
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_name} not found in {folder_path}.")
        
        #Load file
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None

# Assuming folder_path is defined and pointing to the directory containing the files
folder_path = 'D:\credit-score-classification-app-main\models'  # Update this path as needed

model = load_pickle('model.pkl')
scaler = load_pickle('scaler.pkl')

age_default = None
Occupation_default = 0
annual_income_default = 0.00
Monthly_Inhand_Salary_default = 0.00
Num_Bank_Accounts_default = 0
Num_Credit_Card_default =  0
Interest_Rate_default = 0.00
Num_of_Loan_default = 0.00
Delay_from_due_date_default = 0
Num_of_Delayed_Payment_default = 0.00
Changed_Credit_Limit_default = 0.00
Num_Credit_Inquiries_default =0.00
Credit_Mix_default = None
Outstanding_Debt_default  =  0.00
Credit_Utilization_Ratio_default =  0.00
Payment_of_Min_Amount_default = None
Total_EMI_per_month_default  = 0.00
Amount_invested_monthly_default = 0.00
Payment_Behaviour_default = None
Monthly_Balance_default = 0.00
Credit_History_Age_Months_default  = 0 

st.title('Credit Score Prediction')
#st.caption('Modify by HySen ')

#st.markdown('''
 #          I cookeđ this shit while having a fever, so dont expect so much :> Enjoy  
#''')

profile = st.radio('Choose a profile:', options=['A', 'B', 'C'], horizontal=True)
if profile == 'A':
    age_default = 18
    Occupation_default = 1
    annual_income_default = 15000.00
    Monthly_Inhand_Salary_default = 1000.0
    Num_Bank_Accounts_default = 0
    Num_Credit_Card_default = 10
    Interest_Rate_default = 20.0
    Num_of_Loan_default = 3
    Delay_from_due_date_default = 15
    Num_of_Delayed_Payment_default = 5
    Changed_Credit_Limit_default = 1000.0
    Num_Credit_Inquiries_default = 8
    Credit_Mix_default = 0
    Outstanding_Debt_default = 10000.0
    Credit_Utilization_Ratio_default = 90.0
    Payment_of_Min_Amount_default = 0
    Total_EMI_per_month_default = 0.0
    Amount_invested_monthly_default = 0.0
    Payment_Behaviour_default = 0
    Monthly_Balance_default = 200.0
    Credit_History_Age_Months_default = 4
elif profile == 'B':
    age_default = 18
    Occupation_default =  13
    annual_income_default = 500.0
    Monthly_Inhand_Salary_default = 100.0
    Num_Bank_Accounts_default = 0
    Num_Credit_Card_default = 1
    Interest_Rate_default = 20.0
    Num_of_Loan_default = 3
    Delay_from_due_date_default = 15
    Num_of_Delayed_Payment_default = 5
    Changed_Credit_Limit_default = 10.0
    Num_Credit_Inquiries_default = 8
    Credit_Mix_default = 1
    Outstanding_Debt_default = 1500.0
    Credit_Utilization_Ratio_default = 14.0
    Payment_of_Min_Amount_default = 1
    Total_EMI_per_month_default = 0.0
    Amount_invested_monthly_default = 0.0
    Payment_Behaviour_default = 1
    Monthly_Balance_default = 200.0
    Credit_History_Age_Months_default = 4
elif profile == 'C':
    age_default = 45
    Occupation = 3
    annual_income_default = 120.0
    Monthly_Inhand_Salary_default = 1000.0
    Num_Bank_Accounts_default = 3
    Num_Credit_Card_default = 1
    Interest_Rate_default = 5.0
    Num_of_Loan_default = 1
    Delay_from_due_date_default = 30
    Num_of_Delayed_Payment_default = 20
    Changed_Credit_Limit_default = 500.0
    Num_Credit_Inquiries_default = 1
    Credit_Mix_default = 2
    Outstanding_Debt_default = 500.0
    Credit_Utilization_Ratio_default = 20.0
    Payment_of_Min_Amount_default = 1
    Total_EMI_per_month_default = 20.0
    Amount_invested_monthly_default = 10.0
    Payment_Behaviour_default = 0
    Monthly_Balance_default = 500.0
    Credit_History_Age_Months_default = 120

with st.sidebar:
    age = st.slider('What is your age?', min_value=18, max_value=100, step=1, value=age_default)
    Occupation = st.selectbox('What is your Occupation?', ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Lawyer','Developer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager','Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'],index=Occupation_default)
    annual_income = st.number_input('What is your Annual Income?', min_value=0.00, max_value=300000.00, value=annual_income_default)
    Monthly_Inhand_Salary = st.number_input('What is your Monthly Inhand Salary?', min_value=0.00, max_value=25000.00, value=Monthly_Inhand_Salary_default)
    Num_Bank_Accounts = st.number_input('How many bank accounts do you have?', min_value=0, max_value=20, step=1, value=Num_Bank_Accounts_default)
    Num_Credit_Card = st.number_input('How many credit cards do you have?', min_value=0, max_value=12, step=1, value=Num_Credit_Card_default)
    Interest_Rate = st.number_input('What is the interest rate on your loans?', min_value=0.00, max_value=50.00, step=0.1, value=Interest_Rate_default)
    Num_of_Loan = st.number_input('How many loans do you have?', min_value=0, max_value=20, step=1, value=Num_of_Loan_default)
    Delay_from_due_date = st.number_input('How many days delayed from due date?', min_value=0, max_value=365, step=1, value=Delay_from_due_date_default)
    Num_of_Delayed_Payment = st.number_input('How many delayed payments do you have?', min_value=0, max_value=20, step=1, value=Num_of_Delayed_Payment_default)
    Changed_Credit_Limit = st.number_input('How much has your credit limit changed?', min_value=0.00, max_value=100000.00, value=Changed_Credit_Limit_default)
    Num_Credit_Inquiries = st.number_input('How many credit inquiries have you had?', min_value=0, max_value=20, step=1, value=Num_Credit_Inquiries_default)
    Credit_Mix = st.selectbox('What is your Credit Mix?', ['Good', 'Standard', 'Bad'],index=Credit_Mix_default)
    Outstanding_Debt = st.number_input('What is your Outstanding Debt?', min_value=0.00, max_value=1000000.00, value=Outstanding_Debt_default)
    Credit_Utilization_Ratio = st.slider('What is your credit card utilization ratio?', min_value=0.00, max_value=100.00, value=Credit_Utilization_Ratio_default)
    Payment_of_Min_Amount = st.radio('Have you paid the minimum amount on at least one of your credit cards?', ['Yes', 'No'], index=Payment_of_Min_Amount_default)
    Total_EMI_per_month = st.number_input('How much EMI do you pay monthly?', min_value=0.00, max_value=5000.00, value=Total_EMI_per_month_default)
    Amount_invested_monthly = st.number_input('How much do you invest monthly?', min_value=0.00, max_value=5000.00, value=Amount_invested_monthly_default)
    Payment_Behaviour = st.selectbox('What is your Payment Behaviour?', [
       'Low_spent_Medium_value_payments',
       'Low_spent_Small_value_payments',
       'Low_spent_Large_value_payments',
       'High_spent_Small_value_payments',
       'High_spent_Medium_value_payments',
       'High_spent_Large_value_payments'], index=Payment_Behaviour_default)
    Monthly_Balance = st.number_input('What is your Monthly Balance?', min_value=0.00, max_value=5000.00, value=Monthly_Balance_default)
    Credit_History_Age_Months = st.number_input('How many months old is your credit history?', min_value=0, max_value=500, step=1, value=Credit_History_Age_Months_default)

    run = st.button( 'Run the numbers!')

st.header('Credit Score Results')

col1, col2 = st.columns([3, 2])

with col2:
    x1 = [0, 6, 0]
    x2 = [0, 4, 0]
    x3 = [0, 2, 0]
    y = ['0', '1', '2']

    f, ax = plt.subplots(figsize=(5,2))

    p1 = sns.barplot(x=x1, y=y, color='#3EC300')
    p1.set(xticklabels=[], yticklabels=[])
    p1.tick_params(bottom=False, left=False)
    p2 = sns.barplot(x=x2, y=y, color='#FAA300')
    p2.set(xticklabels=[], yticklabels=[])
    p2.tick_params(bottom=False, left=False)
    p3 = sns.barplot(x=x3, y=y, color='#FF331F')
    p3.set(xticklabels=[], yticklabels=[])
    p3.tick_params(bottom=False, left=False)

    plt.text(0.7, 1.05, "POOR", horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.text(2.5, 1.05, "STANDARD", horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.text(4.7, 1.05, "GOOD", horizontalalignment='left', size='medium', color='white', weight='semibold')

    ax.set(xlim=(0, 6))
    sns.despine(left=True, bottom=True)

    figure = st.pyplot(f)

with col1:

    placeholder = st.empty()

    if run:
        resp = {
            'age': age,
            'Occupation': Occupation,
            'annual_income': annual_income,
            'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
            'Num_Bank_Accounts': Num_Bank_Accounts,
            'Num_Credit_Card': Num_Credit_Card,
            'Interest_Rate': Interest_Rate,
            'Num_of_Loan': Num_of_Loan,
            'Delay_from_due_date': Delay_from_due_date,
            'Num_of_Delayed_Payment': Num_of_Delayed_Payment,
            'Changed_Credit_Limit': Changed_Credit_Limit,
            'Num_Credit_Inquiries': Num_Credit_Inquiries,
            'Credit_Mix': Credit_Mix,
            'Outstanding_Debt': Outstanding_Debt,
            'Credit_Utilization_Ratio': Credit_Utilization_Ratio,
            'Payment_of_Min_Amount': Payment_of_Min_Amount,
            'Total_EMI_per_month': Total_EMI_per_month,
            'Amount_invested_monthly': Amount_invested_monthly,
            'Payment_Behaviour': Payment_Behaviour,
            'Monthly_Balance': Monthly_Balance,
            'Credit_History_Age_Months': Credit_History_Age_Months
        }
        output = transform_resp(resp)
        output = pd.DataFrame(output, index=[0])
        output.loc[:, :] = scaler.transform(output)
        
        credit_score = model.predict(output)[0]
        
        
        if credit_score == 2:
            st.balloons()
            t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
            placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
            st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
        elif credit_score == 1:
            t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
            placeholder.markdown('Your credit score is **STANDARD**.')
            st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
        elif credit_score == 0:
            t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
            placeholder.markdown('Your credit score is **POOR**.')
            st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
        plt.gca().add_patch(t1)
        figure.pyplot(f)
        prob_fig, ax = plt.subplots()

        #with st.expander('Click to see how certain the algorithm was'):
         #   plt.pie(model.predict_proba(output)[0], labels=['Poor', 'Regular', 'Good'], autopct='%.0f%%')
          #  st.pyplot(prob_fig)
        
        with st.expander('Click to see how much each feature weight'):
            importance = model.feature_importances_
            # List of feature names
            columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
           'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
           'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
           'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
           'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months',
           'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

            # Create DataFrame with importance values and feature names
            importance_df = pd.DataFrame({
              'Feature': columns,
             'Importance': importance
              })

            # Sort by importance
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            # Plotting the figure
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
            plt.title('Feature Importance in XGBoost Model')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            st.pyplot(fig)