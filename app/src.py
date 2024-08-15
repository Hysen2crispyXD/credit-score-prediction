from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
def transform_resp(resp):
    def yes_no(column):
        if resp[column] == 'Yes':
            return 1
        else:
            return 0
    
    le_occupation = LabelEncoder()
    le_credit_mix = LabelEncoder()
  
    occupations = ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Lawyer','Developer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager','Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']
    credit_mix = ['Good', 'Standard', 'Bad']
    payment_behaviour_order = [
        'Low_spent_Small_value_payments', 
        'Low_spent_Medium_value_payments',
        'Low_spent_Large_value_payments',
        'High_spent_Small_value_payments',
        'High_spent_Medium_value_payments',
        'High_spent_Large_value_payments'
    ]

    le_occupation.fit(occupations)
    le_credit_mix.fit(credit_mix)
    oe_payment_behaviour = OrdinalEncoder(categories=[payment_behaviour_order])

    
    resp['Occupation'] = le_occupation.transform([resp['Occupation']])[0]
    resp['Credit_Mix'] = le_credit_mix.transform([resp['Credit_Mix']])[0]
    resp['Payment_Behaviour'] = oe_payment_behaviour.fit_transform([[resp['Payment_Behaviour']]])[0][0]


    output = {
        'Age': resp['age'],
        'Occupation': resp['Occupation'],
        'Annual_Income': resp['annual_income'],
        'Monthly_Inhand_Salary': resp['Monthly_Inhand_Salary'],
        'Num_Bank_Accounts': resp['Num_Bank_Accounts'],
        'Num_Credit_Card': resp['Num_Credit_Card'],
        'Interest_Rate': resp['Interest_Rate'],
        'Num_of_Loan': resp['Num_of_Loan'],
        'Delay_from_due_date': resp['Delay_from_due_date'],
        'Num_of_Delayed_Payment': resp['Num_of_Delayed_Payment'],
        'Changed_Credit_Limit': resp['Changed_Credit_Limit'],
        'Num_Credit_Inquiries': resp['Num_Credit_Inquiries'],
        'Credit_Mix': resp['Credit_Mix'],
        'Outstanding_Debt': resp['Outstanding_Debt'],
        'Credit_Utilization_Ratio': resp['Credit_Utilization_Ratio'],
        'Payment_of_Min_Amount': yes_no('Payment_of_Min_Amount'),
        'Total_EMI_per_month': resp['Total_EMI_per_month'],
        'Amount_invested_monthly': resp['Amount_invested_monthly'],
        'Payment_Behaviour': resp['Payment_Behaviour'],
        'Monthly_Balance': resp['Monthly_Balance'],
        'Credit_History_Age_Months': resp['Credit_History_Age_Months']
    }

    return output

