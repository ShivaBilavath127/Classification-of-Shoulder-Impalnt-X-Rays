from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
import pymysql
import tkinter as tk


import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from skimage import io, transform
from sklearn import preprocessing
import numpy as np
import joblib
import cv2
from imblearn.over_sampling import SMOTE



model_folder = "model"

# Create model directory if it doesn't exist
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

def uploadDataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")


def DatasetPreprocessing():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder

    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")
    categories_file = os.path.join(model_folder, "categories.txt")
    
    # Save categories to a file for later use
    if not os.path.exists(categories_file):
        # Make sure the model directory exists
        os.makedirs(model_folder, exist_ok=True)
        with open(categories_file, 'w') as f:
            for category in categories:
                f.write(f"{category}\n")
        
    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        print("X and Y arrays loaded successfully.")
    else:
        X = [] # input array
        Y = [] # output array
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(root+"/"+directory[j])
                    img_resized = resize(img_array, (64, 64, 3))
                    # Append the input image array to X
                    X.append(img_resized.flatten())
                    # Append the index of the category in categories list to Y
                    Y.append(categories.index(name))
        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X)
        np.save(Y_file, Y)

    text.insert(END,"Dataset Normalization & Preprocessing Task Completed\n\n")

def Dataset_SMOTE():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    labels, label_count = np.unique(Y, return_counts=True)
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(X, Y)
    labels_resampled, label_count_resampled = np.unique(Y, return_counts=True)
    plt.figure(figsize=(10, 5))

# Before SMOTE
    plt.subplot(1, 2, 1)
    plt.bar(labels, label_count, color='skyblue', alpha=0.8)
    plt.xlabel("Output Type")
    plt.ylabel("Count")
    plt.title("Before SMOTE")

    # After SMOTE
    plt.subplot(1, 2, 2)
    plt.bar(labels_resampled, label_count_resampled, color='lightgreen', alpha=0.8)
    plt.xlabel("Output Type")
    plt.ylabel("Count")
    plt.title("After SMOTE")
    plt.tight_layout()
    plt.show()


def Train_test_splitting():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    global X_train, X_test, y_train, y_test, scaler

 
    #splitting dataset into train and test where application using 80% dataset for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"70% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"30% dataset user for testing   : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, testY, predict):
    global categories
    labels = categories
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    
    # Generate and display confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion Matrix") 
    plt.ylabel('True Class') 
    plt.xlabel('Predicted Class') 
    plt.tight_layout()
    plt.show() 

#now train existing algorithm    
def Existing_Classifier():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    global classifier
    
    text.insert(END, "Loading and evaluating SVM Classifier...\n")
    
    accuracy = []
    precision = []
    recall = [] 
    fscore = []
    
    if os.path.exists('model/SVM_model.pkl'):
        text.insert(END, "Loading existing SVM model from file...\n")
        classifier = joblib.load('model/SVM_model.pkl')
    else:                       
        text.insert(END, "Training new SVM model...\n")
        classifier = SVC(kernel='poly', C=1.0, gamma='scale', random_state=42)
        classifier.fit(X_train, y_train)
        # Save the model to file
        os.makedirs('model', exist_ok=True)
        joblib.dump(classifier, 'model/SVM_model.pkl')
        text.insert(END, "SVM model saved to file.\n")

    # Generate predictions
    text.insert(END, "Evaluating model on test data...\n")
    y_pred_svm = classifier.predict(X_test)
    
    # Calculate and display metrics
    text.insert(END, "Computing performance metrics...\n")
    calculateMetrics("Existing SVM", y_test, y_pred_svm)
    
    text.insert(END, "SVM Classifier evaluation complete!\n")

def Proposed_Classifier():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test
    global classifier
    
    text.insert(END, "Loading and evaluating Random Forest Classifier...\n")
    
    if os.path.exists('model/RFC_Model.pkl'):
        text.insert(END, "Loading existing Random Forest model from file...\n")
        classifier = joblib.load('model/RFC_Model.pkl')
    else:
        text.insert(END, "Training new Random Forest model...\n")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        # Save the model to file
        os.makedirs('model', exist_ok=True)
        joblib.dump(classifier, 'model/RFC_Model.pkl')
        text.insert(END, "Random Forest model saved to file.\n")
    
    # Generate predictions
    text.insert(END, "Evaluating model on test data...\n")
    y_pred_rf = classifier.predict(X_test)
    
    # Calculate and display metrics
    text.insert(END, "Computing performance metrics...\n")
    calculateMetrics("Proposed RFC", y_test, y_pred_rf)
    
    text.insert(END, "Random Forest Classifier evaluation complete!\n")

     
def predict():
    global classifier, categories
    
    # Create testImages directory if it doesn't exist
    test_images_dir = "testImages"
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
        text.insert(END, f"Created {test_images_dir} directory for test images\n")
    
    # Check if classifier is already loaded
    if 'classifier' not in globals() or classifier is None:
        # No classifier loaded, show model selection dialog
        model_choice = simpledialog.askstring("Model Selection", 
                                             "Which model would you like to use? (Enter 'RFC' for Random Forest or 'SVM' for Support Vector Machine)")
        
        if model_choice and model_choice.upper() == 'RFC':
            if os.path.exists('model/RFC_Model.pkl'):
                text.insert(END, "Loading Random Forest Classifier model...\n")
                classifier = joblib.load('model/RFC_Model.pkl')
            else:
                messagebox.showerror("Error", "Random Forest model not found. Please train the model first.")
                return
        elif model_choice and model_choice.upper() == 'SVM':
            if os.path.exists('model/SVM_model.pkl'):
                text.insert(END, "Loading SVM Classifier model...\n")
                classifier = joblib.load('model/SVM_model.pkl')
            else:
                messagebox.showerror("Error", "SVM model not found. Please train the model first.")
                return
        else:
            # Try to load any available model
            if os.path.exists('model/RFC_Model.pkl'):
                text.insert(END, "Loading Random Forest Classifier by default...\n")
                classifier = joblib.load('model/RFC_Model.pkl')
            elif os.path.exists('model/SVM_model.pkl'):
                text.insert(END, "Loading SVM Classifier by default...\n")
                classifier = joblib.load('model/SVM_model.pkl')
            else:
                messagebox.showerror("Error", "No classifier models found. Please train at least one model first.")
                return
    
    # Select an image file for prediction
    filename = filedialog.askopenfilename(initialdir=test_images_dir, 
                                          title="Select an image for prediction",
                                          filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), 
                                                    ("All files", "*.*")))
    if not filename:
        return
        
    try:
        # Read and process the image
        text.insert(END, f"Processing image: {os.path.basename(filename)}\n")
        img = cv2.imread(filename)
        if img is None:
            messagebox.showerror("Error", "Could not read the image file")
            return
        
        # Display original image with predicted class    
        img_resize = resize(img, (64, 64, 3))
        img_preprocessed = [img_resize.flatten()]
        
        # Make prediction
        text.insert(END, "Predicting implant type...\n")
        output_number = classifier.predict(img_preprocessed)[0]
        output_name = categories[output_number]
        
        text.insert(END, f"Prediction result: {output_name}\n")

        # Display the results
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {output_name}", fontsize=14)
        plt.text(10, 20, f'Predicted: {output_name}', color='white', fontsize=14, 
                weight='bold', backgroundcolor='black', alpha=0.7)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during prediction: {str(e)}")
        text.insert(END, f"Error: {str(e)}\n")


def connect_db():
    return pymysql.connect(host='localhost', user='root', password='root', database='sparse_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        global categories
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        # Try to load categories if they're not already loaded
                        if 'categories' not in globals() or not categories:
                            try:
                                # Look for dataset directory
                                dataset_dir = None
                                for dir_name in os.listdir():
                                    if os.path.isdir(dir_name) and "implant" in dir_name.lower():
                                        dataset_dir = dir_name
                                        break
                                
                                if dataset_dir:
                                    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                                    text.delete('1.0', END)
                                    text.insert(END, f"Found categories: {categories}\n")
                                else:
                                    # Try to find categories from the model
                                    X_file = os.path.join(model_folder, "categories.txt")
                                    if os.path.exists(X_file):
                                        with open(X_file, 'r') as f:
                                            categories = f.read().splitlines()
                                        text.delete('1.0', END)
                                        text.insert(END, f"Loaded categories: {categories}\n")
                            except Exception as e:
                                text.delete('1.0', END)
                                text.insert(END, f"Warning: Could not load categories: {str(e)}\n")
                                text.insert(END, "Please make sure to run the preprocessing steps first.\n")
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

def comparison_graph():
    text.delete('1.0', END)
    if len(accuracy) < 2:
        messagebox.showerror("Error", "Please run both classifiers (Existing SVM and Proposed RFC) before comparing")
        return
    
    text.insert(END, "Comparing SVM and Random Forest Classifier Performance\n")
    
    # Define metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F-Score']
    svm_values = [accuracy[0], precision[0], recall[0], fscore[0]]
    rf_values = [accuracy[1], precision[1], recall[1], fscore[1]]
    
    # Create comparison bar chart
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bar1 = plt.bar(x - width/2, svm_values, width, label='SVM', color='skyblue')
    bar2 = plt.bar(x + width/2, rf_values, width, label='Random Forest', color='lightgreen')
    
    plt.title('Performance Comparison: SVM vs Random Forest', fontweight='bold', fontsize=14)
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Score (%)', fontweight='bold')
    plt.xticks(x, metrics)
    plt.ylim(0, 105)  # Setting y-axis limit to 105 for better visualization
    
    # Add values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bar1)
    add_labels(bar2)
    
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Display the results in text widget too
    text.insert(END, "\nPerformance Comparison:\n")
    text.insert(END, f"{'Metric':<12}{'SVM':<15}{'Random Forest':<15}{'Difference':<15}\n")
    text.insert(END, "-" * 60 + "\n")
    
    for i, metric in enumerate(metrics):
        diff = rf_values[i] - svm_values[i]
        text.insert(END, f"{metric:<12}{svm_values[i]:<15.2f}{rf_values[i]:<15.2f}{diff:+<15.2f}\n")
        
    # Indicate which model performed better overall
    svm_avg = sum(svm_values) / len(svm_values)
    rf_avg = sum(rf_values) / len(rf_values)
    
    text.insert(END, "\nOverall Average Performance:\n")
    text.insert(END, f"SVM: {svm_avg:.2f}%\n")
    text.insert(END, f"Random Forest: {rf_avg:.2f}%\n")
    
    if rf_avg > svm_avg:
        text.insert(END, f"\nRandom Forest performs better by {rf_avg - svm_avg:.2f}% on average.\n")
    elif svm_avg > rf_avg:
        text.insert(END, f"\nSVM performs better by {svm_avg - rf_avg:.2f}% on average.\n")
    else:
        text.insert(END, "\nBoth models perform equally on average.\n")

# Admin Button Functions
def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=140, y=200)
    tk.Button(main, text="Preprocessing", command=DatasetPreprocessing, font=font1).place(x=300, y=200)
    tk.Button(main, text="SMOTE", command=Dataset_SMOTE, font=font1).place(x=470, y=200)
    tk.Button(main, text="Train Test Splitting", command=Train_test_splitting, font=font1).place(x=640, y=200)
    tk.Button(main, text="Existing Classifier", command=Existing_Classifier, font=font1).place(x=810, y=200)
    tk.Button(main, text="Proposed Classifier", command=Proposed_Classifier, font=font1).place(x=980, y=200)
    tk.Button(main, text="Compare Models", command=comparison_graph, font=font1).place(x=1150, y=200)

# User Button Functions
def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction", command=predict, font=font1).place(x=550, y=200)
    
    # Add a message to guide users
    text.delete('1.0', END)
    if 'categories' not in globals() or not categories:
        text.insert(END, "Warning: Categories not loaded. Please ask an administrator to process the dataset first.\n")
    else:
        text.insert(END, "Welcome to the Shoulder Implant Classification System!\n\n")
        text.insert(END, "Click the 'Prediction' button to classify a shoulder implant X-ray image.\n")
        text.insert(END, f"The system can identify the following categories: {categories}\n")

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()
    

# Main tkinter window
main = tk.Tk()
main.geometry("1300x1200")
main.title("Machine Learning-Based Classification of Shoulder Implant X-Rays for Manufacturer Identification")

# Title
font = ('times', 18, 'bold')
title = tk.Label(main, text="Machine Learning-Based Classification of Shoulder Implant X-Rays for Manufacturer Identification", bg='white', fg='black', font=font, height=2, width=100)
title.pack()

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=170)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=250)
text.config(font=font1)


# Admin and User Buttons
font1 = ('times', 14, 'bold')


tk.Button(main, text="Hospital Signup", command=lambda: signup("Admin"), font=font1, width=20, height=2, bg='Lightpink').place(x=100, y=100)

tk.Button(main, text="Patient Signup", command=lambda: signup("User"), font=font1, width=20, height=2, bg='Lightpink').place(x=400, y=100)


admin_button = tk.Button(main, text="Hospital Login", command=lambda: login("Admin"), font=font1, width=20, height=2, bg='Lightpink')
admin_button.place(x=700, y=100)

user_button = tk.Button(main, text="Patient Login", command=lambda: login("User"), font=font1, width=20, height=2, bg='Lightpink')
user_button.place(x=1000, y=100)

main.config(bg='Lightblue')
main.mainloop()