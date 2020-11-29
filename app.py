##importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import streamlit as st
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
 
 
def main():
    """Semi Automated ML App with StreamLit"""
    st.title("Semi Auto ML App")
    st.text("Using Streamlit == 0.71.0")
    
    activities = ['EDA', 'Plot', 'Model Building', 'About']
    
    choice = st.sidebar.selectbox("Select Activity", activities) 
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis") 
        data = st.file_uploader("Upload file", type = ["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head()) 
            if st.checkbox("Show shape"):
                st.write(df.shape)
            
            if st.checkbox("Show Columns"):
                all_cols = df.columns.to_list()
                st.write(all_cols)
            
            if st.checkbox("Select columns to show"):
                selected_columns = st.multiselect("Select Columns", all_cols)
                cols = df[selected_columns]
                st.dataframe(cols)  
                
            
            
            
            if st.checkbox("Show Summary"):
                st.write(df.describe())
                
            if st.checkbox("Show Data Types"):
                st.write(df.dtypes)
            
            if st.checkbox("Show Number of target class"):
                st.write(df.iloc[:,-1].value_counts())     
                
            if st.checkbox("Correlation Plot(Matplotlib)"):
                plt.matshow(df.corr())
                st.pyplot()
            
            if st.checkbox("Correlation Plot(Seaborn)"):
                st.write(sns.heatmap(df.corr(), annot = True))
                st.pyplot() 
                
            if st.checkbox("Pie Plot"):
                all_columns = df.columns.to_list()
                cols_to_plot = st.selectbox("Select 1 column", all_columns)
                pie_plot = df[cols_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
                st.write(pie_plot)
                st.pyplot() 
                
                
    elif choice == 'Plot':
        st.subheader("Data Visualization")
        
        data = st.file_uploader("Upload file", type = ["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head()) 
        
            if st.checkbox("Show value counts:"):
                st.write(df.iloc[:,-1].value_counts().plot(kind = 'bar'))
                st.pyplot() 
                
        ###Customizable Plots
        all_cols_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot", all_cols_names)
        
        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))
            
            ##Plot by Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)
            
            elif type_of_plot == "bar":
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)
            
            elif type_of_plot == "line":
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)
            #Custom Plot
            elif type_of_plot:
                cust_plot= df[selected_columns_names].plot(kind = type_of_plot)
                st.write(cust_plot)
                st.pyplot() 
                
    elif choice == "Model Building":
        st.subheader("Building ML Model")
        data = st.file_uploader("Upload file", type = ["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head()) 
            
            #Model Building 
            X = df.iloc[:, : -1]
            y = df.iloc[:, -1]
            seed = 42    
            
            ##Prepareing Models
            models = []
            models.append(('LR', LogisticRegression()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("Trees",DecisionTreeClassifier()))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC()))
            
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            for name, model in models:
                kfold = model_selection.KFold(n_splits = 10, random_state = seed)
                cv_results = model_selection.cross_val_score(model, X,y,cv = kfold, scoring = scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"Model_name":name, "Model_accuracy": cv_results.mean(), "Standara Deviation": cv_results.std()}
                all_models.append(accuracy_results)
               
            if st.checkbox("Metric As Table: "):
                st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std), columns = ['Algorithm', 'Mean of Accuracy', 'Std']))
                
            if st.checkbox("Metric As JSON"):
                st.json(all_models)
            
            
    elif choice == "About":
        st.subheader("About")
        
if __name__ == '__main__':
    main() 