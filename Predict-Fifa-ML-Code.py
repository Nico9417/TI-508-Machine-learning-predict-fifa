# Import libraries needed 
import pandas as pd
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

# Import data 
file_path = 'Predict-Fifa-ML-Dataset.csv'
data = pd.read_csv(file_path)

# Drop space around positions
data['Position'] = data['Position'].str.strip()

# Drop goalkeepers (GK) from the data
data = data[data['Position'] != 'GK']

#####################################################################
# Positions mapping

positions = {
    "ATT": ["LW", "RW", "ST", "LM", "RM"],
    "MID": ["CM", "CDM", "CAM"],
    "DEF": ["LB", "RB", "CB"]
}

# Function to map positions
def map_position(position):
    for category, values in positions.items():
        if position in values:
            return category
    return position  # If the position is not in the dictionary, it is kept unchanged
   

data['Position'] = data['Position'].apply(map_position)

# Define features with more detailed stats than the basic ones
features = [
    'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 'Acceleration', 'Sprint Speed',
    'Positioning', 'Finishing', 'Shot Power', 'Long Shots', 'Volleys', 'Penalties',
    'Vision', 'Crossing', 'Free Kick Accuracy', 'Short Passing', 'Long Passing',
    'Curve', 'Dribbling', 'Agility', 'Balance', 'Reactions', 'Ball Control',
    'Composure', 'Interceptions', 'Heading Accuracy', 'Def Awareness',
    'Standing Tackle', 'Sliding Tackle', 'Jumping', 'Stamina', 'Strength',
    'Aggression'
]

# Drop rows with missing values for any of the selected features
data = data.dropna(subset=features)

# Define features and label
X = data[features] 
y = data['Position']

# Standardize the features 
scaler = Normalizer()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train multiple models and store them for future use
# We decided to use the following models: 
# KNN, 
# Random Forest, 
# SVM, 
# Logistic Regression
k_number=12
Random_state_number = 42
models = {
    'KNN': KNeighborsClassifier(n_neighbors=k_number),
    'Random Forest': RandomForestClassifier(n_estimators=210, random_state=Random_state_number),
    'SVM': SVC(kernel='linear', probability=True),
    'Logistic Regression (Softmax)': LogisticRegression(max_iter=150, multi_class='multinomial', solver='lbfgs'),
    'Bagging SVC Classifier': BaggingClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=210, random_state=42).fit(X_train, y_train)
}

# Function to train and evaluate each model
def evaluate_models(models, X_train, X_test, y_train, y_test):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {model_name}: {accuracy * 100:.2f}%")
        print("\n")
        print("Classification report:\n", classification_report(y_test, y_pred))
        print("\n" + "-"*50 + "\n")
                
        if model_name == 'Logistic Regression (Softmax)':
            # Binarize the output
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            n_classes = y_test_bin.shape[1]

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])

            # Plot ROC curve for each class
            plt.figure()
            colors = ['aqua', 'darkorange', 'cornflowerblue']
            pos = ['ATT', 'DEF', 'MID']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of class {pos[i]} (area = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for multi-class')
            plt.legend(loc="lower right")
            plt.show()
                

# Function to determine the optimal K for KNN
def find_best_k(X_train, y_train, X_test, y_test):
    Ks = 100
    mean_acc = np.zeros((Ks-1))

    for n in range(1, Ks):
        neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    # Plot accuracy vs K
    plt.plot(range(1, Ks), mean_acc, 'g')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbors (K)')
    plt.title('Accuracy vs. Number of Neighbors (K)')
    plt.show()

    print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# Function to predict player's position based on their name and chosen model
def predict_player_position(player_name, model, model_name, data, features, scaler):
    # Case-insensitive search for player's name
    player_data = data[data['Name'].str.contains(player_name, case=False, na=False)]
    
    if player_data.empty:
        print("Player not found!")
        return
    
    # Extract the player's features
    player_features = player_data[features].values
    
    # Normalize the player's features using the same scaler as the training data
    player_features_scaled = scaler.transform(player_features)
    
    # Predict the player's position using the selected model
    predicted_position = model.predict(player_features_scaled)
    
    # Extract first prediction if the result is an array
    predicted_position = predicted_position[0] if len(predicted_position) > 0 else predicted_position

    print(f"The predicted position for {player_name} with {model_name} model is: {predicted_position}")


# Main code
if __name__ == "__main__":
    # Training the models before making predictions
    for model_name, model in models.items():
        model.fit(X_train, y_train)

    def print_menu():
        print("\n")
        print("-----------------------------")
        print("- Welcome in our project - : ")
        print("-----------------------------")
        print("\n")
        print("1 - Search the position of the player")
        print("2 - Evaluate models")
        print("3 - Plot confusion matrices")
        print("4 - Exit")
        print("\n")

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == '4':
            find_best_k(X_train, y_train, X_test, y_test)
            break
        elif choice == '1':
            while True:
                player_name = input("Enter the player's name to predict their position (or type 'exit' to quit): ").strip()

                if player_name.lower() == 'exit':
                    print("Exiting the program. Goodbye!")
                    break

                # Validate the player's name
                player_data = data[data['Name'].str.contains(player_name, case=False, na=False)]

                if not player_data.empty:
                    print(f"Player '{player_name}' found. Proceeding to model selection.")
                    break  # Exit loop if player is found
                else:
                    print("Player not found! Please enter a valid player's name.")

            # If the user has chosen to exit, break out of the main loop
            if player_name.lower() == 'exit':
                break

            # Display the menu for model selection
            print("Select a model for prediction:")
            print("1 - Use KNN")
            print("2 - Use Random Forest")
            print("3 - Use SVM")
            print("4 - Use Logistic Regression (Softmax)")
            print("5 - Find best value of k for KNN")

            # Secure input for model selection
            model_choice = None
            while model_choice not in ['1', '2', '3', '4', '5']:
                model_choice = input("Enter the number corresponding to the model: ").strip()
                if model_choice == '1' :
                    k_number = input("How many neighbors do you want to test:").strip()         
                    break
                if model_choice == '2' :
                    Random_state_number = input("Put the random_state hyperparameter: ").strip()
                    break 
                if model_choice not in ['1', '2', '3', '4', '5']:
                    print("Invalid choice. Please select a number between 1 and 5.")

            # Execute the chosen option
            if model_choice == '5':
                find_best_k(X_train, y_train, X_test, y_test)
            else:
                # Mapping choices to their respective models
                model_mapping = {
                    '1': models['KNN'],
                    '2': models['Random Forest'],
                    '3': models['SVM'],
                    '4': models['Logistic Regression (Softmax)']
                }
                model_name_mapping = {
                    '1': 'KNN',
                    '2': 'Random Forest',
                    '3': 'SVM',
                    '4': 'Logistic Regression (Softmax)'
                }
                
                # Récupération du modèle sélectionné et de son nom
                selected_model = model_mapping[model_choice]
                selected_model_name = model_name_mapping[model_choice]
                
                # Appel à la prédiction avec les bons arguments
                predict_player_position(player_name, selected_model, selected_model_name, data, features, scaler)
        elif choice == '2':
            evaluate_models(models, X_train, X_test, y_train, y_test)
        elif choice == '3':
            # Plot confusion matrix for each model
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix for {model_name}')
                plt.colorbar()
                tick_marks = np.arange(len(set(y)))
                plt.xticks(tick_marks, set(y), rotation=45)
                plt.yticks(tick_marks, set(y))
                plt.ylabel('True label')
                plt.xlabel('Predicted label')

                # Add text annotations
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.show()
        else:
            print("Invalid choice. Please select a valid option.")
            