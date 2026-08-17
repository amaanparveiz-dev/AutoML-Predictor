import pandas as pd
import numpy as np
import threading
import time
import os
from customtkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
import warnings
warnings.filterwarnings("ignore")

# Appearance and Theme
set_appearance_mode("dark")
set_default_color_theme("blue")

# Global Variables
path = None
df = None
selected_output_column = None
selected_input_columns = []
checkbox_vars = []
checkbox_widgets = []
best_model = None
best_split = None
best_score = -1
best_estimators=0
best_alpha = 0
best_neighbors = 0
x= None
y =None
step = 0 
is_output_selection = False  
encoders = {}  
original_df = None 
predict_vars = []  
predict_widgets = []   


App = CTk()
App.title("FAA Data Predictor")
App.geometry("1000x700")
App.minsize(800, 600)

classifier = BooleanVar(value=False) 

def Clean_Data():
    global df , selected_input_columns , selected_output_column, encoders, original_df

    
    selected_columns = selected_input_columns + [selected_output_column]
    df = df[selected_columns]   

    filtered_df = df.copy()

    for col in df.select_dtypes(include='object').columns:
        value_percents = df[col].value_counts(normalize=True) * 100
        valid_values = value_percents[value_percents > 2.5].index
        filtered_df = filtered_df[filtered_df[col].isin(valid_values)]

    df = filtered_df

    show_status_message("Outliers Removed")
    time.sleep(1)

    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 1:
            print(col)
            df.drop(col, axis=1, inplace=True)
            if col in selected_input_columns:
                selected_input_columns.remove(col)
                show_status_message("Removed Column : " + str(col))
                time.sleep(1)
                    
                    
    original_df = df.copy()
            

    for col in df.select_dtypes(include=['int64','float64','int','float']).columns:
        df[col] = df[col].fillna(df[col].mean())

    columns_to_encode = df.select_dtypes(include='object').columns
    encoders = {}

    for col in columns_to_encode :
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le 

    show_status_message("Label Encoder Implemented")
    time.sleep(1)

    for col in df.select_dtypes(include=['int64','float64','int','float']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    show_status_message("Null Values Filled (If Any)")
    time.sleep(1)
    

def Check_Models():
    global best_score, best_model , best_split , best_estimators , best_alpha , best_neighbors ,  x , y

    x = df[selected_input_columns]
    y = df[selected_output_column]


    if(classifier.get()):
        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 1
                best_split = i
            print(model.score(x_test, y_test))
        print("Done 1")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = DecisionTreeClassifier().fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 2
                best_split = i
            print(model.score(x_test, y_test))
        print("Done 2")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = RandomForestClassifier(n_estimators=i*10, random_state=42).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 3
                best_split = i
                best_estimators = i
            print(model.score(x_test, y_test))
        print("Done 3")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = RidgeClassifier(alpha=i*0.2).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 4
                best_split = i
                best_alpha = i
            print(model.score(x_test, y_test))
        print("Done 4")

        for i in range(1, 11):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 5
                best_split = i
                best_neighbors = i
            print(model.score(x_test, y_test))
        print("Done 5")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = GradientBoostingClassifier(n_estimators=i*10, learning_rate=0.1, max_depth=3, random_state=42).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 6
                best_split = i
                best_estimators = i
            print(model.score(x_test, y_test))
        print("Done 6")
        print(best_model)

    else:
        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = LinearRegression().fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 7
                best_split = i
                print("Best Score Updated " )

            print(model.score(x_test, y_test))
        print("Done 7")
        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = DecisionTreeRegressor().fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 8
                best_split = i
                print("Best Score Updated " )

            print(model.score(x_test, y_test))
        print("Done 8")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = RandomForestRegressor(n_estimators=i*10 , random_state=42).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 9
                best_split = i
                best_estimators = i
                print("Best Score Updated ")
            print(model.score(x_test, y_test))
        print("Done 9")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = Ridge(alpha=i*0.2).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 10
                best_split = i
                best_alpha = i
                print("Best Score Updated ")

            print(model.score(x_test, y_test))
        print("Done 10")      

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = Lasso(alpha=i*0.2).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 11
                best_alpha = i
                print("Best Score Updated ")

            print(model.score(x_test, y_test))
        print("Done 11")

        for i in range(1, 19):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i * 0.05, random_state=42)
            model = GradientBoostingRegressor(n_estimators=i*10, learning_rate=0.1, max_depth=3, random_state=42).fit(x_train, y_train)
            if model.score(x_test, y_test) > best_score:
                best_score = model.score(x_test, y_test)
                best_model = 12
                best_split = i
                print("Best Score Updated ") 
                best_estimators = i
            print(model.score(x_test, y_test))
        print("Done 12")

        print(best_model)
        print(best_score)

def Train_Data():
    global df, step
    Clean_Data()
    print(df.head())
    show_status_message("Data Cleaned Successfully")
    Check_Models()
    show_status_message("Data Trained Successfully")
    time.sleep(2)
    show_status_message("Model Accuracy = " + str(best_score))
    step += 1
    update_progress()
    action_btn.configure(text="Predict Data", command=Predict_Data)

def Predict_Data():
    global predict_vars, predict_widgets
    
    # Create prediction window
    predict_window = CTkToplevel(App)
    predict_window.title("Make Prediction")
    predict_window.geometry("800x600")
    
    # Main frame
    main_predict_frame = CTkFrame(master=predict_window, corner_radius=15)
    main_predict_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    # Title
    CTkLabel(master=main_predict_frame, text="Make Prediction", font=("Arial", 24)).pack(pady=10)
    
    # Input frame
    input_frame = CTkScrollableFrame(master=main_predict_frame)
    input_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Clear previous widgets
# Clear previous widgets safely
    for widget in predict_widgets:
        try:
            if isinstance(widget, CTkOptionMenu):
                # Manually remove trace before destroying
                widget._variable.trace_remove('write', widget._variable_callback_name)
            widget.destroy()
        except Exception as e:
            print(f"Warning: Could not destroy widget - {str(e)}")

        predict_vars = []
        predict_widgets = []
        
    # Create input fields for each feature
    for i, col in enumerate(selected_input_columns):
        row_frame = CTkFrame(master=input_frame, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)
        
        CTkLabel(master=row_frame, text=f"{col}:", width=120).pack(side="left", padx=5)
        
        # Check if column was encoded
        if col in encoders:
            # Get original categories
            categories = original_df[col].unique()
            var = StringVar(value=str(categories[0]))
            optionmenu = CTkOptionMenu(master=row_frame, variable=var, values=[str(c) for c in categories])
            optionmenu.pack(side="left", fill="x", expand=True)
            predict_widgets.append(optionmenu)
        else:
            # Numerical input
            var = StringVar(value="0")
            entry = CTkEntry(master=row_frame, textvariable=var)
            entry.pack(side="left", fill="x", expand=True)
            predict_widgets.append(entry)
        
        predict_vars.append(var)
    
    # Result label
    result_label = CTkLabel(master=main_predict_frame, text="", font=("Arial", 16))
    result_label.pack(pady=10)
    
    # Prediction button
    def make_prediction():
        try:
            # Prepare input data
            input_data = {}
            for i, col in enumerate(selected_input_columns):
                value = predict_vars[i].get()
                
                # Convert categorical values back to encoded numbers
                if col in encoders:
                    # Handle unseen categories
                    try:
                        encoded_value = encoders[col].transform([value])[0]
                    except ValueError:
                        # If value wasn't in original data, use most common category
                        encoded_value = df[col].mode()[0]
                    input_data[col] = encoded_value
                else:
                    # Convert numerical values
                    try:
                        input_data[col] = float(value)
                    except ValueError:
                        input_data[col] = 0.0
            
            # Create DataFrame from input data
            input_df = pd.DataFrame([input_data])
            
            # Select the best model based on previous training
            if classifier.get():  # Classification
                if best_model == 1:
                    model = LogisticRegression(max_iter=1000)
                elif best_model == 2:
                    model = DecisionTreeClassifier()
                elif best_model == 3:
                    model = RandomForestClassifier(n_estimators=best_estimators*10, random_state=42)
                elif best_model == 4:
                    model = RidgeClassifier(alpha=best_alpha*0.2)
                elif best_model == 5:
                    model = KNeighborsClassifier(n_neighbors=best_neighbors)
                elif best_model == 6:
                    model = GradientBoostingClassifier(n_estimators=best_estimators*10, learning_rate=0.1, max_depth=3, random_state=42)
            else:  # Regression
                if best_model == 7:
                    model = LinearRegression()
                elif best_model == 8:
                    model = DecisionTreeRegressor()
                elif best_model == 9:
                    model = RandomForestRegressor(n_estimators=best_estimators*10, random_state=42)
                elif best_model == 10:
                    model = Ridge(alpha=best_alpha*0.2)
                elif best_model == 11:
                    model = Lasso(alpha=best_alpha*0.2)
                elif best_model == 12:
                    model = GradientBoostingRegressor(n_estimators=best_estimators*10, learning_rate=0.1, max_depth=3, random_state=42)
            
            # Train the model with best parameters
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=best_split*0.05, random_state=42)
            model.fit(x_train, y_train)
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # For classification, get original class names
            if classifier.get() and selected_output_column in encoders:
                prediction = encoders[selected_output_column].inverse_transform(prediction)
            
            result_label.configure(text=f"Predicted Output: {prediction[0]}", text_color="#4cc9f0")
            
        except Exception as e:
            result_label.configure(text=f"Error: {str(e)}", text_color="red")
    
    predict_button = CTkButton(master=main_predict_frame, text="Predict", command=make_prediction)
    predict_button.pack(pady=10)

# Main Frame
main_frame = CTkFrame(master=App, corner_radius=15)
main_frame.pack(padx=40, pady=40, fill="both", expand=True)

# Header Frame
header_frame = CTkFrame(master=main_frame, fg_color="transparent")
header_frame.pack(pady=(10, 20), fill="x")

title_label = CTkLabel(master=header_frame, text="FAA Data Predictor", font=("Arial", 32, "bold"))
title_label.pack(pady=(0, 5))

desc_label = CTkLabel(master=header_frame, text="Select output and input columns from your CSV data", text_color="gray70", font=("Arial", 14))
desc_label.pack()

# Progress Frame
progress_frame = CTkFrame(master=main_frame, fg_color="transparent")
progress_frame.pack(pady=(10, 10), fill="x", padx=20)

progress_steps = ["Load Data", "Select Model Type", "Select Output", "Select Inputs", "Train Data", "Predict Data"]
progress_labels = []

for i, step_text in enumerate(progress_steps):
    step_inner_frame = CTkFrame(master=progress_frame, fg_color="transparent")
    step_inner_frame.pack(side="left", expand=True)

    circle = CTkLabel(master=step_inner_frame, text=str(i+1), width=30, height=30, corner_radius=15, font=("Arial", 14, "bold"), fg_color="#2b2b2b" if i > 0 else "#3a7ebf")
    circle.pack()

    label = CTkLabel(master=step_inner_frame, text=step_text, font=("Arial", 14), text_color="gray70" if i > 0 else "white")
    label.pack(pady=(5, 0))
    progress_labels.append((circle, label))

    if i < len(progress_steps) - 1:
        arrow = CTkLabel(master=progress_frame, text="➞", font=("Arial", 20), text_color="gray70")
        arrow.pack(side="left")

# Progress Bar
linear_progress = CTkProgressBar(master=main_frame, height=10)
linear_progress.set(0.0)
linear_progress.pack(fill="x", padx=20, pady=(0, 20))

# Content Frame
content_frame = CTkFrame(master=main_frame, fg_color="transparent")
content_frame.pack(fill="both", expand=True, padx=20, pady=10)

instructions_label = CTkLabel(master=content_frame, text="Please select a CSV file to begin", font=("Arial", 16), text_color="gray70")
instructions_label.pack(pady=(10, 20))

checkbox_container = CTkFrame(master=content_frame, fg_color="transparent")
checkbox_container.pack(fill="both", expand=True, pady=(0, 20))

checkbox_scroll_frame = CTkScrollableFrame(master=checkbox_container, orientation="horizontal", height=60, fg_color="#2b2b2b", corner_radius=10)
checkbox_scroll_frame.pack(fill="x", padx=10)

# Classifier/Regression Toggle
classifier_toggle = CTkCheckBox(master=checkbox_scroll_frame, text="Classifier Mode (unchecked = Regression)", variable=classifier)
classifier_toggle.pack(side="left", padx=10, pady=10)

preview_frame = CTkFrame(master=content_frame, fg_color="#1f1f1f", corner_radius=10)
preview_frame.pack(fill="x", padx=10, pady=10)

preview_title = CTkLabel(master=preview_frame, text="Selected Columns", font=("Arial", 14, "bold"), anchor="w")
preview_title.pack(padx=15, pady=(10, 5), anchor="w")

output_preview = CTkLabel(master=preview_frame, text="Output: None selected", font=("Arial", 14), text_color="gray70", anchor="w")
output_preview.pack(padx=15, pady=(0, 5), anchor="w")

input_preview = CTkLabel(master=preview_frame, text="Inputs: None selected", font=("Arial", 14), text_color="gray70", anchor="w")
input_preview.pack(padx=15, pady=(0, 10), anchor="w")

# Status Bar
status_frame = CTkFrame(master=main_frame, height=30, fg_color="transparent")
status_frame.pack(fill="x", padx=20, pady=(0, 10))

status_label = CTkLabel(master=status_frame, text="Ready", font=("Arial", 20, "bold"),text_color="gray70", anchor="center")
status_label.pack(fill="x", expand=True)


def show_status_message(text="✔ Step Completed", color="green"):
    status_label.configure(text=text, text_color=color, font=("Arial", 20, "bold"))
    App.update()
    time.sleep(2)
    if "✔" in text or "❌" in text:
        status_label.configure(text="Ready", text_color="gray70", font=("Arial", 20, "bold"))

def update_progress():
    total_steps = 5  # Now 5 steps with Predict Data
    progress_fraction = step / total_steps
    linear_progress.set(progress_fraction)

    for i, (circle, label) in enumerate(progress_labels):
        if i <= step:
            circle.configure(fg_color="#3a7ebf")
            label.configure(text_color="white")
        else:
            circle.configure(fg_color="#2b2b2b")
            label.configure(text_color="gray70")

def update_preview():
    output_preview.configure(text=f"Output: {selected_output_column}" if selected_output_column else "Output: None selected", text_color="#4cc9f0" if selected_output_column else "gray70")
    if selected_input_columns:
        input_text = ", ".join(selected_input_columns[:3])
        if len(selected_input_columns) > 3:
            input_text += f" (+{len(selected_input_columns)-3} more)"
        input_preview.configure(text=f"Inputs: {input_text}", text_color="#4cc9f0")
    else:
        input_preview.configure(text="Inputs: None selected", text_color="gray70")

def handle_output_check(index):
    def callback():
        if is_output_selection:
            for i, var in enumerate(checkbox_vars):
                if i != index:
                    var.set(0)
        update_preview()
    return callback

def clear_checkboxes():
    for widget in checkbox_widgets:
        widget.destroy()
    checkbox_vars.clear()
    checkbox_widgets.clear()

def next_step():
    global step, path, df, selected_output_column, selected_input_columns, is_output_selection

    if step == 0:
       
        path = filedialog.askopenfilename(title="Select a CSV File", filetypes=[["CSV Files", "*.csv"]])

        if path:
            try:
                df = pd.read_csv(path)
                App.after(0, lambda: show_status_message("✔ Data loaded successfully"))
                step += 1
                update_progress()
                action_btn.configure(text="Confirm Model Type")
                
                # Show model type selection after loading data
                instructions_label.configure(text="Select model type (Classification or Regression)", text_color="#4cc9f0")
                classifier_toggle.pack(side="left", padx=10, pady=10)
                
            except Exception as e:
                App.after(0, lambda: show_status_message(f"❌ Error loading file: {str(e)}", color="red"))

    elif step == 1:
        # Confirm model type selection
        clear_checkboxes()
        is_output_selection = True
        
        # Get the model type from the checkbox
        model_type = "Classification" if classifier.get() else "Regression"
        App.after(0, lambda: show_status_message(f"✔ Model type selected: {model_type}"))
        
        # Prepare for output column selection
        for i, col in enumerate(df.columns):
            var = IntVar()
            chk = CTkCheckBox(master=checkbox_scroll_frame, text=col, variable=var, command=handle_output_check(i), border_width=1, checkbox_width=20, checkbox_height=20, corner_radius=5)
            chk.pack(side="left", padx=10, pady=10)
            checkbox_vars.append(var)
            checkbox_widgets.append(chk)

        instructions_label.configure(text="Select ONE output column (target variable)", text_color="#4cc9f0")
        step += 1
        update_progress()
        action_btn.configure(text="Confirm Output Selection")
        update_preview()
        test_btn.configure(state="normal")
        
        # Hide classifier toggle after selection
        classifier_toggle.pack_forget()

    elif step == 2:
        selected = [chk.cget("text") for var, chk in zip(checkbox_vars, checkbox_widgets) if var.get() == 1]
        if len(selected) == 1:
            selected_output_column = selected[0]
            clear_checkboxes()
            is_output_selection = False

            for col in df.columns:
                if col != selected_output_column:
                    var = IntVar()
                    chk = CTkCheckBox(master=checkbox_scroll_frame, text=col, variable=var, border_width=1, checkbox_width=20, checkbox_height=20, corner_radius=5)
                    chk.pack(side="left", padx=10, pady=10)
                    checkbox_vars.append(var)
                    checkbox_widgets.append(chk)

            instructions_label.configure(text="Select input columns (features)", text_color="#4cc9f0")
            App.after(0, lambda: show_status_message(f"✔ Output selected: {selected_output_column}"))
            step += 1
            update_progress()
            action_btn.configure(text="Confirm Input Selection")
            update_preview()
        else:
            App.after(0, lambda: show_status_message("❌ Please select exactly one output column", color="red"))

    elif step == 3:
        selected_input_columns.clear()
        selected_input_columns.extend([chk.cget("text") for var, chk in zip(checkbox_vars, checkbox_widgets) if var.get() == 1])
        if selected_input_columns:
            instructions_label.configure(text="Selection complete! Ready to train.", text_color="#90ee90")
            App.after(0, lambda: show_status_message(f"✔ {len(selected_input_columns)} input columns selected"))
            step += 1
            update_progress()
            action_btn.configure(text="Train Model", command=Train_Data)
            update_preview()
        else:
            App.after(0, lambda: show_status_message("❌ Please select at least one input column", color="red"))

def reset_app():
    global step, path, df, selected_output_column, selected_input_columns, is_output_selection
    step = 0
    path = None
    df = None
    selected_output_column = None
    selected_input_columns = []
    clear_checkboxes()
    instructions_label.configure(text="Please select a CSV file to begin", text_color="gray70")
    update_progress()
    linear_progress.set(0.0)
    action_btn.configure(text="Open CSV File", command=next_step)
    test_btn.configure(state="disabled")
    classifier_toggle.pack(side="left", padx=10, pady=10)  # Show again on reset
    update_preview()
    show_status_message("✔ Application reset", "green")

# Button Frame
button_frame = CTkFrame(master=main_frame, fg_color="transparent")
button_frame.pack(pady=(10, 20), fill="x", padx=20)

action_btn = CTkButton(master=button_frame, text="Open CSV File", command=next_step, font=("Arial", 14, "bold"), height=40, fg_color="#3a7ebf", hover_color="#2d6399")
action_btn.pack(side="left", padx=(0, 10), fill="x", expand=True)

test_btn = CTkButton(master=button_frame, text="Reset", command=reset_app, font=("Arial", 14, "bold"), height=40, fg_color="#4CAF50", hover_color="#3d8b40", state="disabled")
test_btn.pack(side="left", padx=(10, 0), fill="x", expand=True)

update_preview()
App.mainloop()