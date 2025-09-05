import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("Covid Data Uncleaned.csv")


print(df.info())  # Get information about datatypes and nulls

print(df.head())  

df.drop_duplicates(inplace=True)   


# --- Data Cleaning and Transformation ---

# Replacing 0,1 and 97 with appropriate value 
df["SEX"] = df["SEX"].replace({1: "F", 2: "M"})
df["USMER"]=df["USMER"].replace({1:"First level",2:"Second level",3:"Third level"})
df["PATIENT_TYPE"]=df["PATIENT_TYPE"].replace({1:"Returned Home",2:"Hospitalized"})


df["DATE_DIED"]=pd.to_datetime(df["DATE_DIED"],errors="coerce").dt.date   # Convert 'DATE_DIED' from object to datetime

df["DATE_DIED"].fillna("Alive",inplace=True)   # Filling missing values


cols_to_clean = ["INTUBED","PNEUMONIA", "PREGNANT", "DIABETES", "COPD", "ASTHMA", "INMSUPR",
                 "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY",
                 "RENAL_CHRONIC", "TOBACCO", "ICU"]

for col in cols_to_clean:
    df[col] = df[col].replace({1:"Yes", 2:"No", 97:"Not Specified", 98:"Not Specified", 99:"Not Specified"})





df["CLASIFFICATION_FINAL"]=pd.cut(x=df["CLASIFFICATION_FINAL"],bins=[0,3,7],labels=["Positive","Negative"])  # Apply a custom classification for 'CLASIFFICATION_FINAL'

df['Patient_Condition']=df['DATE_DIED']
df['Patient_Condition'][df['DATE_DIED']=='9999-99-99']='Alive'
df['Patient_Condition'][df['DATE_DIED']!='9999-99-99']='Dead'

df["Age-Group"]=pd.cut(df["AGE"],bins=[1,18,50,150],labels=["Minor","Adult","Senior"])

print(df.isnull().sum())


print(df.head())  # Display first few rows to verify changes

print(df.info())


# --- Data Visualization ---





# Data distribution based on Medical Unit.
a=sns.countplot(x=df["MEDICAL_UNIT"],label=[1,2,3,4,5,6,7,8,9,10,11,12,13])
a.bar_label(a.containers[0])
plt.title("Data distribution based on Medical Unit")
plt.legend(loc='best')
plt.show()



#COVID-19 Classification by SEX Status.
b=sns.countplot(x="SEX",data=df,palette=["blue","orange"],hue="CLASIFFICATION_FINAL")
b.bar_label(b.containers[0])
plt.title("COVID-19 Classification by SEX Status")
plt.xlabel("SEX")
plt.ylabel("COUNT")
plt.legend(loc='best')
plt.show()


# Distribution by PATIENT_TYPE

d=sns.countplot(x=df["PATIENT_TYPE"],label=["Yes","No"],data=df)
d.bar_label(d.containers[0])
plt.title("Distribution by PATIENT_TYPE")
plt.legend(loc='best')
plt.show()



# COVID-19 Classification by AGE-GROUP

e=sns.countplot(x=df["Age-Group"],data=df,hue="CLASIFFICATION_FINAL")
e.bar_label(e.containers[0])
plt.legend(loc="best")
plt.xlabel("AGE_GROUP")
plt.ylabel("COUNT")
plt.title("COVID-19 Classification by AGE-GROUP")
plt.show()




#COVID-19 Classification by Diabetes Status.

c=sns.countplot(x="DIABETES",data=df,hue="CLASIFFICATION_FINAL")
plt.title("COVID-19 Classification by Diabetes Status")
plt.xlabel("Diabetes Status")
c.bar_label(c.containers[0])
plt.legend(loc='best')
plt.show()

df.to_csv("Covid Data cleaned.csv",index=False)