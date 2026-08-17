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

# ============================================================
# THEME / DESIGN TOKENS  (visual layer only — no logic here)
# ============================================================
set_appearance_mode("dark")
set_default_color_theme("blue")

BG = "#0A0E14"
SURFACE = "#10151F"
SURFACE_ALT = "#161C29"
SURFACE_ROW = "#1B2231"
BORDER = "#242C3D"
INPUT_BG = "#1B2231"

ACCENT = "#14B8A6"
ACCENT_HOVER = "#0D9488"
ACCENT_SOFT = "#5EEAD4"

SUCCESS = "#34D399"
DANGER = "#F87171"

TEXT_PRIMARY = "#EDF1F7"
TEXT_SECONDARY = "#8A93A6"
TEXT_MUTED = "#4E5768"

FONT_FAMILY = "Segoe UI"

def F(size, weight="normal"):
    return (FONT_FAMILY, size, weight)

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
App.title("AutoML Predictor")
App.geometry("1150x780")
App.minsize(940, 640)
App.configure(fg_color=BG)

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

    show_status_message("Outliers removed", auto_reset=False)
    time.sleep(1)

    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 1:
            print(col)
            df.drop(col, axis=1, inplace=True)
            if col in selected_input_columns:
                selected_input_columns.remove(col)
                show_status_message("Removed column: " + str(col), auto_reset=False)
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

    show_status_message("Label encoder applied", auto_reset=False)
    time.sleep(1)

    for col in df.select_dtypes(include=['int64','float64','int','float']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    show_status_message("Null values filled (if any)", auto_reset=False)
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
    show_status_message("Data cleaned successfully", auto_reset=False)
    Check_Models()
    show_status_message("Data trained successfully", auto_reset=False)
    time.sleep(2)
    show_status_message("Model accuracy = " + str(best_score), auto_reset=False)
    step += 1
    update_progress()
    action_btn.configure(text="Predict Data", command=Predict_Data)

def Predict_Data():
    global predict_vars, predict_widgets
    
    # Create prediction window
    predict_window = CTkToplevel(App)
    predict_window.title("Make Prediction")
    predict_window.geometry("860x660")
    predict_window.configure(fg_color=BG)
    predict_window.minsize(680, 520)
    
    # Main frame
    main_predict_frame = CTkFrame(master=predict_window, corner_radius=18, fg_color=SURFACE, border_width=1, border_color=BORDER)
    main_predict_frame.pack(padx=26, pady=26, fill="both", expand=True)
    
    # Title
    title_block = CTkFrame(master=main_predict_frame, fg_color="transparent")
    title_block.pack(fill="x", padx=30, pady=(28, 4), anchor="w")
    CTkLabel(master=title_block, text="MAKE A PREDICTION", font=F(14, "bold"), text_color=ACCENT, anchor="w").pack(anchor="w")
    CTkLabel(master=title_block, text="Enter feature values to generate an output", font=F(15), text_color=TEXT_SECONDARY, anchor="w").pack(anchor="w", pady=(4, 0))

    CTkFrame(master=main_predict_frame, height=1, fg_color=BORDER).pack(fill="x", padx=30, pady=(18, 0))
    
    # Input frame
    input_frame = CTkScrollableFrame(master=main_predict_frame, fg_color="transparent")
    input_frame.pack(fill="both", expand=True, padx=26, pady=(16, 10))
    
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
        row_frame = CTkFrame(master=input_frame, fg_color=SURFACE_ROW, corner_radius=10)
        row_frame.pack(fill="x", pady=5, padx=2)
        
        CTkLabel(master=row_frame, text=col, width=170, font=F(14, "bold"), text_color=TEXT_PRIMARY, anchor="w").pack(side="left", padx=(16, 8), pady=13)
        
        # Check if column was encoded
        if col in encoders:
            # Get original categories
            categories = original_df[col].unique()
            var = StringVar(value=str(categories[0]))
            optionmenu = CTkOptionMenu(master=row_frame, variable=var, values=[str(c) for c in categories],
                                        fg_color=INPUT_BG, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                                        dropdown_fg_color=SURFACE_ROW, corner_radius=8, font=F(14),
                                        text_color=TEXT_PRIMARY)
            optionmenu.pack(side="left", fill="x", expand=True, padx=(0, 16), pady=13)
            predict_widgets.append(optionmenu)
        else:
            # Numerical input
            var = StringVar(value="0")
            entry = CTkEntry(master=row_frame, textvariable=var, fg_color=INPUT_BG, corner_radius=8,
                              border_width=1, border_color=BORDER, font=F(14), text_color=TEXT_PRIMARY)
            entry.pack(side="left", fill="x", expand=True, padx=(0, 16), pady=13)
            predict_widgets.append(entry)
        
        predict_vars.append(var)
    
    # Result label
    result_label = CTkLabel(master=main_predict_frame, text="", font=F(17, "bold"))
    result_label.pack(pady=(4, 6))
    
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
            
            result_label.configure(text=f"Predicted Output:  {prediction[0]}", text_color=ACCENT_SOFT)
            
        except Exception as e:
            result_label.configure(text=f"Error: {str(e)}", text_color=DANGER)
    
    predict_button = CTkButton(master=main_predict_frame, text="Run Prediction", command=make_prediction,
                                font=F(15, "bold"), height=46, corner_radius=10,
                                fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#04211D")
    predict_button.pack(pady=(0, 26), padx=30, fill="x")

# ============================================================
# MAIN WINDOW LAYOUT — sidebar + content
# ============================================================

outer = CTkFrame(master=App, fg_color=BG)
outer.pack(fill="both", expand=True)

shell = CTkFrame(master=outer, corner_radius=20, fg_color=SURFACE, border_width=1, border_color=BORDER)
shell.pack(padx=24, pady=24, fill="both", expand=True)
shell.grid_columnconfigure(0, weight=0)
shell.grid_columnconfigure(1, weight=1)
shell.grid_rowconfigure(0, weight=1)

# ---------------- Sidebar ----------------
sidebar = CTkFrame(master=shell, width=250, corner_radius=16, fg_color=SURFACE_ALT)
sidebar.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
sidebar.grid_propagate(False)

brand_block = CTkFrame(master=sidebar, fg_color="transparent")
brand_block.pack(fill="x", padx=22, pady=(26, 4))
CTkLabel(master=brand_block, text="AUTOML PREDICTOR", font=F(16, "bold"), text_color=TEXT_PRIMARY, anchor="w").pack(anchor="w")
CTkLabel(master=brand_block, text="Machine Learning Studio", font=F(12), text_color=TEXT_MUTED, anchor="w").pack(anchor="w", pady=(2, 0))

CTkFrame(master=sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=22, pady=(18, 20))

steps_block = CTkFrame(master=sidebar, fg_color="transparent")
steps_block.pack(fill="x", padx=22)

progress_steps = ["Load Data", "Select Model Type", "Select Output", "Select Inputs", "Train Data", "Predict Data"]
progress_labels = []

for i, step_text in enumerate(progress_steps):
    row = CTkFrame(master=steps_block, fg_color="transparent")
    row.pack(fill="x", pady=7)

    circle = CTkLabel(master=row, text=str(i+1), width=26, height=26, corner_radius=13,
                       font=F(13, "bold"), fg_color=ACCENT if i == 0 else INPUT_BG,
                       text_color="#04211D" if i == 0 else TEXT_MUTED)
    circle.pack(side="left")

    label = CTkLabel(master=row, text=step_text, font=F(14), anchor="w",
                      text_color=TEXT_PRIMARY if i == 0 else TEXT_SECONDARY)
    label.pack(side="left", padx=(12, 0))

    progress_labels.append((circle, label))

sidebar_spacer = CTkFrame(master=sidebar, fg_color="transparent")
sidebar_spacer.pack(fill="both", expand=True)

progress_bottom = CTkFrame(master=sidebar, fg_color="transparent")
progress_bottom.pack(fill="x", padx=22, pady=(0, 24))
CTkLabel(master=progress_bottom, text="OVERALL PROGRESS", font=F(11, "bold"), text_color=TEXT_MUTED, anchor="w").pack(anchor="w", pady=(0, 8))
linear_progress = CTkProgressBar(master=progress_bottom, height=6, corner_radius=6,
                                  fg_color=INPUT_BG, progress_color=ACCENT)
linear_progress.set(0.0)
linear_progress.pack(fill="x")

# ---------------- Content ----------------
content = CTkFrame(master=shell, fg_color="transparent")
content.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)

content_header = CTkFrame(master=content, fg_color="transparent")
content_header.pack(fill="x", padx=8, pady=(10, 18))
instructions_label = CTkLabel(master=content_header, text="Please select a CSV file to begin",
                               font=F(21, "bold"), text_color=TEXT_PRIMARY, anchor="w")
instructions_label.pack(anchor="w")
CTkLabel(master=content_header, text="Follow the steps on the left to configure and train your model",
          font=F(14), text_color=TEXT_SECONDARY, anchor="w").pack(anchor="w", pady=(4, 0))

checkbox_container = CTkFrame(master=content, fg_color="transparent")
checkbox_container.pack(fill="both", expand=False, padx=8, pady=(0, 16))

checkbox_scroll_frame = CTkScrollableFrame(master=checkbox_container, orientation="horizontal", height=72,
                                            fg_color=SURFACE_ALT, corner_radius=14,
                                            border_width=1, border_color=BORDER)
checkbox_scroll_frame.pack(fill="x")

# Classifier/Regression Toggle
classifier_toggle = CTkCheckBox(master=checkbox_scroll_frame, text="Classifier Mode (unchecked = Regression)",
                                 variable=classifier, font=F(14), text_color=TEXT_PRIMARY,
                                 fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=TEXT_MUTED,
                                 checkbox_width=20, checkbox_height=20, corner_radius=5)
classifier_toggle.pack(side="left", padx=16, pady=18)

preview_frame = CTkFrame(master=content, fg_color=SURFACE_ALT, corner_radius=14,
                          border_width=1, border_color=BORDER)
preview_frame.pack(fill="x", padx=8, pady=(0, 8))

preview_title = CTkLabel(master=preview_frame, text="SELECTED COLUMNS", font=F(12, "bold"), text_color=TEXT_MUTED, anchor="w")
preview_title.pack(padx=20, pady=(16, 6), anchor="w")

output_preview = CTkLabel(master=preview_frame, text="Output: None selected", font=F(15), text_color=TEXT_SECONDARY, anchor="w")
output_preview.pack(padx=20, pady=(0, 6), anchor="w")

input_preview = CTkLabel(master=preview_frame, text="Inputs: None selected", font=F(15), text_color=TEXT_SECONDARY, anchor="w")
input_preview.pack(padx=20, pady=(0, 16), anchor="w")

content_spacer = CTkFrame(master=content, fg_color="transparent")
content_spacer.pack(fill="both", expand=True)

# Status Bar
status_frame = CTkFrame(master=content, height=44, fg_color=SURFACE_ALT, corner_radius=12,
                         border_width=1, border_color=BORDER)
status_frame.pack(fill="x", padx=8, pady=(0, 14))

status_label = CTkLabel(master=status_frame, text="Ready", font=F(15, "bold"), text_color=TEXT_SECONDARY, anchor="center")
status_label.pack(fill="x", expand=True, pady=11)


def show_status_message(text="Step completed", color="green", auto_reset=True):
    display_color = {"green": SUCCESS, "red": DANGER}.get(color, color)
    status_label.configure(text=text, text_color=display_color, font=F(15, "bold"))
    App.update()
    time.sleep(2)
    if auto_reset:
        status_label.configure(text="Ready", text_color=TEXT_SECONDARY, font=F(15, "bold"))

def update_progress():
    total_steps = 5  # Now 5 steps with Predict Data
    progress_fraction = step / total_steps
    linear_progress.set(progress_fraction)

    for i, (circle, label) in enumerate(progress_labels):
        if i <= step:
            circle.configure(fg_color=ACCENT, text_color="#04211D")
            label.configure(text_color=TEXT_PRIMARY)
        else:
            circle.configure(fg_color=INPUT_BG, text_color=TEXT_MUTED)
            label.configure(text_color=TEXT_SECONDARY)

def update_preview():
    output_preview.configure(text=f"Output:  {selected_output_column}" if selected_output_column else "Output: None selected", text_color=ACCENT_SOFT if selected_output_column else TEXT_SECONDARY)
    if selected_input_columns:
        input_text = ", ".join(selected_input_columns[:3])
        if len(selected_input_columns) > 3:
            input_text += f" (+{len(selected_input_columns)-3} more)"
        input_preview.configure(text=f"Inputs:  {input_text}", text_color=ACCENT_SOFT)
    else:
        input_preview.configure(text="Inputs: None selected", text_color=TEXT_SECONDARY)

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
                App.after(0, lambda: show_status_message("Data loaded successfully"))
                step += 1
                update_progress()
                action_btn.configure(text="Confirm Model Type")
                
                # Show model type selection after loading data
                instructions_label.configure(text="Select model type (Classification or Regression)", text_color=ACCENT_SOFT)
                classifier_toggle.pack(side="left", padx=16, pady=18)
                
            except Exception as e:
                App.after(0, lambda: show_status_message(f"Error loading file: {str(e)}", color="red"))

    elif step == 1:
        # Confirm model type selection
        clear_checkboxes()
        is_output_selection = True
        
        # Get the model type from the checkbox
        model_type = "Classification" if classifier.get() else "Regression"
        App.after(0, lambda: show_status_message(f"Model type selected: {model_type}"))
        
        # Prepare for output column selection
        for i, col in enumerate(df.columns):
            var = IntVar()
            chk = CTkCheckBox(master=checkbox_scroll_frame, text=col, variable=var, command=handle_output_check(i),
                               border_width=1, checkbox_width=20, checkbox_height=20, corner_radius=5,
                               font=F(14), text_color=TEXT_PRIMARY, fg_color=ACCENT,
                               hover_color=ACCENT_HOVER, border_color=TEXT_MUTED)
            chk.pack(side="left", padx=16, pady=18)
            checkbox_vars.append(var)
            checkbox_widgets.append(chk)

        instructions_label.configure(text="Select ONE output column (target variable)", text_color=ACCENT_SOFT)
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
                    chk = CTkCheckBox(master=checkbox_scroll_frame, text=col, variable=var,
                                       border_width=1, checkbox_width=20, checkbox_height=20, corner_radius=5,
                                       font=F(14), text_color=TEXT_PRIMARY, fg_color=ACCENT,
                                       hover_color=ACCENT_HOVER, border_color=TEXT_MUTED)
                    chk.pack(side="left", padx=16, pady=18)
                    checkbox_vars.append(var)
                    checkbox_widgets.append(chk)

            instructions_label.configure(text="Select input columns (features)", text_color=ACCENT_SOFT)
            App.after(0, lambda: show_status_message(f"Output selected: {selected_output_column}"))
            step += 1
            update_progress()
            action_btn.configure(text="Confirm Input Selection")
            update_preview()
        else:
            App.after(0, lambda: show_status_message("Please select exactly one output column", color="red"))

    elif step == 3:
        selected_input_columns.clear()
        selected_input_columns.extend([chk.cget("text") for var, chk in zip(checkbox_vars, checkbox_widgets) if var.get() == 1])
        if selected_input_columns:
            instructions_label.configure(text="Selection complete — ready to train", text_color=SUCCESS)
            App.after(0, lambda: show_status_message(f"{len(selected_input_columns)} input columns selected"))
            step += 1
            update_progress()
            action_btn.configure(text="Train Model", command=Train_Data)
            update_preview()
        else:
            App.after(0, lambda: show_status_message("Please select at least one input column", color="red"))

def reset_app():
    global step, path, df, selected_output_column, selected_input_columns, is_output_selection
    step = 0
    path = None
    df = None
    selected_output_column = None
    selected_input_columns = []
    clear_checkboxes()
    instructions_label.configure(text="Please select a CSV file to begin", text_color=TEXT_PRIMARY)
    update_progress()
    linear_progress.set(0.0)
    action_btn.configure(text="Open CSV File", command=next_step)
    test_btn.configure(state="disabled")
    classifier_toggle.pack(side="left", padx=16, pady=18)  # Show again on reset
    update_preview()
    show_status_message("Application reset", "green")

# Button Frame
button_frame = CTkFrame(master=content, fg_color="transparent")
button_frame.pack(fill="x", padx=8, pady=(0, 8))

action_btn = CTkButton(master=button_frame, text="Open CSV File", command=next_step, font=F(15, "bold"),
                        height=46, corner_radius=10, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                        text_color="#04211D")
action_btn.pack(side="left", padx=(0, 12), fill="x", expand=True)

test_btn = CTkButton(master=button_frame, text="Reset", command=reset_app, font=F(15, "bold"),
                      height=46, corner_radius=10, fg_color="transparent", hover_color=SURFACE_ROW,
                      text_color=TEXT_SECONDARY, border_width=1, border_color=BORDER, state="disabled")
test_btn.pack(side="left", padx=(12, 0), fill="x", expand=True)

update_preview()
App.mainloop()