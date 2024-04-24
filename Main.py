import streamlit as st
import time 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split



def main():

    st.set_page_config(page_title='Car Price Prediction', page_icon=':car:', layout='centered')

    page_bg_image = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://wallpapercave.com/wp/wp3672140.jpg");
    }

    </style>
    """
    st.markdown(page_bg_image, unsafe_allow_html=True)

    
    st.write(f"<h1 style='color : Violet; text-align:center; padding-bottom :80px;'>Churn Prediction</h3>",unsafe_allow_html=True)
    #https://th.bing.com/th/id/R.1ee114adf3f6f5d36fa03d98dd027595?rik=Z0uB2FeHhylfbQ&riu=http%3a%2f%2fwww.pixelstalk.net%2fwp-content%2fuploads%2f2016%2f06%2fLight-Blue-HD-Backgrounds-Free-Download.jpg&ehk=A1oL99oIPNjoWoX7%2fJZhOrBh8TTdeatDhhq26OJaxJU%3d&risl=&pid=ImgRaw&r=0
    #https://wallpapercave.com/wp/wp7952905.jpg
    st.write(f"<h1 style='color : Black; text-align:center; padding-bottom :80px;'>For Telecom Industries</h3>",unsafe_allow_html=True)



    #Loading the Dataset
    data=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    #Converting the Data into Categorical

    # Replaing the Prediction where 1 indicates Yes and 0 indicates NO

    data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    data['Churn'].replace(to_replace='No', value=0, inplace=True)

    # Replacing the Gender where 1 indicates male and 0 indicates Female

    data['gender'].replace(to_replace='Male',value=1,inplace=True)
    data['gender'].replace(to_replace='Female',value=0,inplace=True)

    #Replacing the Partner where 1 indicates yes and O indicates No

    data['Partner'].replace(to_replace='Yes',value=1,inplace=True)
    data['Partner'].replace(to_replace='No',value=0,inplace=True)

    # Replacing the Dependents where 1 indicates yes and O Indicates No

    data['Dependents'].replace(to_replace='Yes',value=1,inplace=True)
    data['Dependents'].replace(to_replace='No',value=0,inplace=True)

    #Replacing the Phoneservice where 1 indictes yes and 0 indicates No

    data['PhoneService'].replace(to_replace='Yes',value=1,inplace=True)
    data['PhoneService'].replace(to_replace='No',value=0,inplace=True)

    #Replacing the Mutiplines values Where 1 indicates yes and 0 indicates No and No Phone Service Indicates 0

    data['MultipleLines'].replace(to_replace='Yes',value=1,inplace=True)
    data['MultipleLines'].replace(to_replace='No',value=0,inplace=True)
    data['MultipleLines'].replace(to_replace='No phone service',value=0,inplace=True)

    #Replacing the Internet Service Where 1 indicates the DSl and Fiber optic and 0 indicates the No

    data['InternetService'].replace(to_replace='DSL',value=1,inplace=True)
    data['InternetService'].replace(to_replace='Fiber optic',value=1,inplace=True)
    data['InternetService'].replace(to_replace='No',value=0,inplace=True)

    #Replacing the Online Security Where 1 indicates Yes and 0 Indicates No

    data['OnlineSecurity'].replace(to_replace='Yes',value=1,inplace=True)
    data['OnlineSecurity'].replace(to_replace='No',value=0,inplace=True)
    data['OnlineSecurity'].replace(to_replace='No internet service',value=0,inplace=True)

    #Replacing the Device Protection Where 1 indicates Yes and 0 Indicates No

    data['DeviceProtection'].replace(to_replace='Yes',value=1,inplace=True)
    data['DeviceProtection'].replace(to_replace='No',value=0,inplace=True)
    data['DeviceProtection'].replace(to_replace='No internet service',value=0,inplace=True)

    #Replacing the TechSupport Where 1 indicates Yes and 0 Indicates No


    data['TechSupport'].replace(to_replace='Yes',value=1,inplace=True)
    data['TechSupport'].replace(to_replace='No',value=0,inplace=True)
    data['TechSupport'].replace(to_replace='No internet service',value=0,inplace=True)

    #Replacing the StreamingTV Where 1 indicates Yes and 0 Indicates No

    data['StreamingTV'].replace(to_replace='Yes',value=1,inplace=True)
    data['StreamingTV'].replace(to_replace='No',value=0,inplace=True)
    data['StreamingTV'].replace(to_replace='No internet service',value=0,inplace=True)

    #Replacing the StreamingMovies Where 1 indicates Yes and 0 Indicates No

    data['StreamingMovies'].replace(to_replace='Yes',value=1,inplace=True)
    data['StreamingMovies'].replace(to_replace='No',value=0,inplace=True)
    data['StreamingMovies'].replace(to_replace='No internet service',value=0,inplace=True)

    #Replacing the PaperlessBilling Where 1 indicates Yes and 0 Indicates No

    data['PaperlessBilling'].replace(to_replace='Yes',value=1,inplace=True)
    data['PaperlessBilling'].replace(to_replace='No',value=0,inplace=True)

    #Replacing the PaymentMethod Where 1 indicates Electronic check and Mailed check and 2 indicates Bank transfer and Credit card

    data['PaymentMethod'].replace(to_replace='Electronic check',value=1,inplace=True)
    data['PaymentMethod'].replace(to_replace='Mailed check',value=1,inplace=True)
    data['PaymentMethod'].replace(to_replace='Bank transfer (automatic)',value=2,inplace=True)
    data['PaymentMethod'].replace(to_replace='Credit card (automatic)',value=2,inplace=True)

    #Replacing the Contract Where 1 indicates Month-to-month, 2 indicates One year and 3 indicates Two year

    data['Contract'].replace(to_replace='Month-to-month',value=1,inplace=True)
    data['Contract'].replace(to_replace='One year',value=2,inplace=True)
    data['Contract'].replace(to_replace='Two year',value=3,inplace=True)

    #Replacing the Online Backup Where 1 indicates Yes and 0 Indicates No

    data['OnlineBackup'].replace(to_replace='Yes',value=1,inplace=True)
    data['OnlineBackup'].replace(to_replace='No',value=0,inplace=True)
    data['OnlineBackup'].replace(to_replace='No internet service',value=0,inplace=True)

    #Replace the Total Charges with 0 where there is no data
    data['TotalCharges'].replace(to_replace=' ',value=0,inplace=True)
    data['TotalCharges']=data['TotalCharges'].astype(np.float64)


    #Copying the Customer ID in a seperate Variable
    customerID=data['customerID']
    data.drop('customerID',axis=1,inplace=True)

    #Saving the customerID in a csv file
    customerID.to_csv('customerID.csv',index=False)

    #Splitting the Data into X and y

    X=data.drop('Churn',axis=1)
    y=data['Churn']

    #Splitting the Data into Train and Test By using Sklearn
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    #Create a Logistic Regression Model
    logistic_model=LogisticRegression()

    #Train the model
    logistic_model.fit(X_train,y_train)

    #Make Prediction on the Test Data
    y_pred=logistic_model.predict(X_test)

    st.subheader("Please Enter the Details to Predict the Churn")

    #Feature 1
    gender=st.selectbox("Gender",('Male','Female'))
    if gender=='Male':
        gender=1
    else:
        gender=0

    #Feature 2
    senior_citizen=st.selectbox("Senior Citizen",('Yes','No'))
    if senior_citizen=='Yes':
        senior_citizen=1
    else:
        senior_citizen=0

    #Feature 3
    partner=st.selectbox("Partner",('Yes','No'))
    if partner=='Yes':
        partner=1
    else:
        partner=0

    #Feature 4
    #It Is Not So Important Feature So It is better to Exclude it
    dependents=0

    #Feature 5
    Phone_service=st.selectbox("Phone Service",('Yes','No'))
    if Phone_service=='Yes':
        Phone_service=1
    else:
        Phone_service=0

    #Feature 6
    Multiple_lines=st.selectbox("Multiple Lines",('Yes','No'))
    if Multiple_lines=='Yes':
        Multiple_lines=1
    else:
        Multiple_lines=0

    #Feature 7
    Internet_service=st.selectbox("Internet Service",('DSL','Fiber optic','No'))
    if Internet_service=='DSL':
        Internet_service=1
    elif Internet_service=='Fiber optic':
        Internet_service=1
    else:
        Internet_service=0

    #Feature 8
    Online_security=st.selectbox("Online Security : Whether the customer has online security or not",('Yes','No'))
    if Online_security=='Yes':
        Online_security=1
    else:
        Online_security=0

    #Feature 9
    Device_protection=st.selectbox("Device Protection",('Yes','No'))
    if Device_protection=='Yes':
        Device_protection=1
    else:
        Device_protection=0

    #Feature 10
    Tech_support=st.selectbox("Tech Support",('Yes','No'))
    if Tech_support=='Yes':
        Tech_support=1
    else:
        Tech_support=0

    #Feature 11
    StreamingTV=st.selectbox("Streaming TV",('Yes','No'))
    if StreamingTV=='Yes':
        StreamingTV=1
    else:
        StreamingTV=0
    
    #Feature 12
    StreamingMovies=st.selectbox("Streaming Movies",('Yes','No'))
    if StreamingMovies=='Yes':
        StreamingMovies=1
    else:
        StreamingMovies=0
    
    #Feature 13
    paperless_billing=st.selectbox("Paperless Billing",('Yes','No'))
    if paperless_billing=='Yes':
        paperless_billing=1
    else:
        paperless_billing=0

    #Feature 14
    payment_method=st.selectbox("Payment Method",('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'))
    if payment_method=='Electronic check':
        payment_method=1
    elif payment_method=='Mailed check':
        payment_method=1
    elif payment_method=='Bank transfer (automatic)':
        payment_method=2
    else:
        payment_method=2

    #Feature 15
    contract=st.selectbox("Contract",('Month-to-month','One year','Two year'))
    if contract=='Month-to-month':
        contract=1
    elif contract=='One year':
        contract=2
    else:
        contract=3
    
    #Feature 16
    online_backup=st.selectbox("Online Backup : Whether the customer has online backup or not ",('Yes','No'))
    if online_backup=='Yes':
        online_backup=1
    else:
        online_backup=0

    #Feature 17
    tenure=st.number_input("Tenure : Number of months the customer has stayed with the company",min_value=0,max_value=100)

    #Feature 18
    monthly_charges=st.number_input("Monthly Charges : The amount charged to the customer monthly in Rupees ",min_value=0.0,max_value=1000.0)

    #Feature 19
    total_charges=st.number_input("Total Charges : The total amount charged to the customer in  Rupees ",min_value=0.0,max_value=20000.0)

    #Make Prediction on the Input Data
    X_pred=np.array([gender,senior_citizen,partner,dependents,tenure,Phone_service,Multiple_lines,Internet_service,Online_security,online_backup,Device_protection,Tech_support,StreamingTV,StreamingMovies,contract,paperless_billing,payment_method,monthly_charges,total_charges]).reshape(1,-1)
    y_original_pred=0


    if st.button("Predict"):
        with st.spinner('Predicting ...'):
            time.sleep(3)
        y_original_pred=logistic_model.predict(X_pred)

    if y_original_pred<=0.5:
        st.warning("The Customer will Churn")
        st.info("NOTE :These Predictions are  based on the Previous Data ",icon='⚠️')
        time.sleep(3)
        st.toast("Copy Rights ©️ Ajay0304 ",icon='🟥')
        time.sleep(8)

    else:
        st.balloons()
        st.success("The Customer will not Churn")
        st.info("NOTE :These Predictions are  based on the Previous Data",icon='⚠️')
        time.sleep(3)
        st.toast("Copy Rights ©️ Ajay0304 ",icon='🟥')
        time.sleep(8)

    #Evaluate the model
    accuracy=accuracy_score(y_test,y_pred)
    st.write(f"Accuracy : ",accuracy*100)

if __name__=='__main__':
    main()
