import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Charger votre ensemble de données
# Supposez que vous avez un fichier CSV nommé "house_prices.csv"
df = pd.read_csv("https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv")

# Sélectionnez les fonctionnalités et la cible
features_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                     'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                     'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
                     'long', 'sqft_living15', 'sqft_lot15']

X = df[features_columns]
y = df["price"]

# Divisez les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez un modèle de régression linéaire
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Créez un modèle RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Interface utilisateur Streamlit
st.title("Application de prédiction des prix des maisons")
st.image("pricehouse.jpg")

st.sidebar.write('')
st.sidebar.markdown("Made with :smiley: by [Jouini Zied](https://www.linkedin.com/in/zied-jouini/)")
# Entrée utilisateur dans la barre latérale
st.sidebar.title("Entrée les valeurs")
user_input = {}
for feature in features_columns:
    user_input[feature] = st.sidebar.number_input(f"Saisissez la valeur de {feature}", min_value=0)

# Prédiction basée sur les caractéristiques sélectionnées
linear_prediction_button = st.button("Prédire le prix (Régression linéaire)")
rf_prediction_button = st.button("Prédire le prix (Random Forest)")

# Affichage des prédictions dans la page principale
if linear_prediction_button:
    user_input_df = pd.DataFrame([user_input])
    linear_prediction = linear_model.predict(user_input_df)
    st.success(f"Le prix prédit de la maison (Régression linéaire) est {linear_prediction[0]}")

if rf_prediction_button:
    user_input_df = pd.DataFrame([user_input])
    rf_prediction = rf_model.predict(user_input_df)
    st.success(f"Le prix prédit de la maison (Random Forest) est {rf_prediction[0]}")