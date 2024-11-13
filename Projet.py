'''

    TODO : 
    - REFAIRE LA FONCTION MAIN ELLE NE VA PAS DU TOUT 
    - AMÉLIORER L'ACCURACY DES MODÈLES ELLE EST TROP BASSE (ON DOIT TENDRE VERS 100 % OU PAS LOIN)
    - FAIRE DES PLOTS 
    - VOIR CE QU'ELLE A DEMANDÉ DANS LES CONSIGNES (PROJECT.PDF)

'''

# Import libraries needed 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

# Import data 
file_path = './male_players.csv'
data = pd.read_csv(file_path)

# Nettoyage des données : retirer les espaces autour des positions
data['Position'] = data['Position'].str.strip()

# Exclure les gardiens de but (GK) des données
data = data[data['Position'] != 'GK']

############################################ Rassemblement des positions #

positions = {
    "ATT": ["LW", "RW", "ST"],
    "MID": ["CM", "CDM", "CAM", "LM", "RM"],
    "DEF": ["LB", "RB", "CB"]
}

# Fonction pour mapper les positions
def map_position(position):
    for category, values in positions.items():
        if position in values:
            return category
    return position  # Si la position n'est pas dans le dictionnaire, on la garde inchangée


# Define features with more detailed stats than the base ones
features = [
    'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY',               # Base stats on cards
    'Finishing', 'Heading Accuracy', 'Positioning',        # Attacking statistics 
    'Short Passing', 'Long Passing', 'Vision',             # Midfield statistics
    'Ball Control', 'Standing Tackle', 'Sliding Tackle',   # Defensive statistics
    'Interceptions', 'Acceleration', 'Sprint Speed',       # Additional recommended stats
    'Agility', 'Balance', 'Stamina', 'Strength'            # Physical and agility stats 
]
'''
Features supprimé:
'''

# Drop rows with missing values for any of the selected features
data = data.dropna(subset=features)

data['Position'] = data['Position'].apply(map_position)

# Define features and label
X = data[features] 
y = data['Position']

# Normalize the features 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models and store them for future use
models = {
    'KNN': KNeighborsClassifier(n_neighbors=51),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True),
    'Logistic Regression (Softmax)': LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')
}

# Function to train and evaluate each model
def evaluate_models(models, X_train, X_test, y_train, y_test):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {model_name}: {accuracy * 100:.2f}%")
        print("Classification report:\n", classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("\n" + "-"*50 + "\n")

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
def predict_player_position(player_name, model, data, features, scaler):
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
    
    # Simplify predicted positions to broader categories
    if predicted_position in ["ST", "LW", "RW"]:
        predicted_position = "ATT"
    elif predicted_position in ["CM", "CDM", "CAM", "LM", "RM"]:
        predicted_position = "MID"
    elif predicted_position in ["LB", "RB", "CB"]:
        predicted_position = "DEF"

    print(f"The predicted position for {player_name} is: {predicted_position}")


# Main code
if __name__ == "__main__":
    # Evaluate models (décommenter cette ligne pour évaluer les modèles au démarrage)
    # evaluate_models(models, X_train, X_test, y_train, y_test)

    # Find best k for KNN 
    # find_best_k(X_train, y_train, X_test, y_test)

    # Training the models before making predictions
    for model_name, model in models.items():
        model.fit(X_train, y_train)

    while True:  
        # Prompt user to select a player name for prediction
        player_name = input("Enter the player's name to predict their position (or type 'exit' to quit): ")

        if player_name.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break  # End if user input 'exit'

        # Validate the player's name in a loop until a valid name is provided
        while True:
            player_data = data[data['Name'].str.contains(player_name, case=False, na=False)]
            
            if not player_data.empty:
                print(f"Player '{player_name}' found. Proceeding to model selection.")
                break  # Exit loop if player is found
            else:
                player_name = input("Player not found! Please enter a valid player's name (or type 'exit' to quit): ")
                if player_name.lower() == 'exit':
                    print("End of program!")
                    break  # Exit if user input 'exit'

        # If the user has chosen to exit, break out of the main loop
        if player_name.lower() == 'exit':
            break

        # Display the menu for model selection
        print("Select a model for prediction:")
        print("1 - KNN")
        print("2 - Random Forest")
        print("3 - SVM")
        print("4 - Logistic Regression (Softmax)")
        print("5 - Find best value of k for KNN")
        print("6 - Evaluate models")

        # Secure input for model selection
        model_choice = None
        while model_choice not in ['1', '2', '3', '4', '5', '6']:
            model_choice = input("Enter the number corresponding to the model: ")
            if model_choice not in ['1', '2', '3', '4', '5', '6']:
                print("Invalid choice. Please select a number between 1 and 6.")

        # Execute the chosen option
        if model_choice == '5':
            find_best_k(X_train, y_train, X_test, y_test)
        elif model_choice == '6':
            evaluate_models(models, X_train, X_test, y_train, y_test)
        else:
            # Mapping choices to their respective models
            model_mapping = {
                '1': models['KNN'],
                '2': models['Random Forest'],
                '3': models['SVM'],
                '4': models['Logistic Regression (Softmax)']
            }
            selected_model = model_mapping[model_choice]
            predict_player_position(player_name, selected_model, data, features, scaler)
