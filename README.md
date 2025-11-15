⚡ EV Price Predictor
A simple web application built with Streamlit to predict the price of an electric vehicle (EV) based on its key specifications.

The app uses a pre-trained RandomForestRegressor model to estimate the price and includes an interactive dashboard to explore how changing a single feature (like range or battery size) affects the predicted value.

(Add a screenshot of your running app.py here)

Features
Price Prediction: Get an instant price estimate based on 6 key EV specifications.

Interactive Controls: Use sliders and number inputs to adjust:

🔋 Battery Capacity (kWh)

🚀 0-100 km/h (sec)

🪑 Number of Seats

🏎️ Top Speed (km/h)

🛣️ Range (km)

⚡ Efficiency (Wh/km)

Feature-Price Explorer: An interactive Matplotlib graph shows how the predicted price changes as you vary one feature, while holding all others constant.

Project Structure
.
├── app.py              # The main Streamlit web application
├── model.pkl           # The pre-trained scikit-learn pipeline (Scaler + Model)
├── train_model.py      # Script to train and save the model
├── requirements.txt    # Python dependencies
└── cars_data_cleaned.csv # (Required for training - not included)

🚀 How to Run the App
Follow these steps to get the application running on your local machine.

1. Clone the Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

2. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

4. Run the Streamlit App
Now, you can launch the application.

streamlit run app.py

A new tab should automatically open in your web browser, pointing to http://localhost:8501.

🧠 (Optional) How to Re-train the Model
The included model.pkl file is already trained. However, if you want to re-train the model (e.g., if you get new data), you can follow these steps.

1. Get the Data
Make sure you have the dataset named cars_data_cleaned.csv in the same directory. This file is required by train_model.py.

(Note: This dataset is not included in this repository.)

2. Run the Training Script
Execute the train_model.py script. This will load the CSV, train a new RandomForestRegressor pipeline (including a StandardScaler), and save the new model over the old model.pkl file.

python train_model.py

After the script finishes, you can restart your Streamlit app to use the newly trained model.

🛠️ Libraries Used
Streamlit: For building the interactive web app.

Pandas: For data manipulation.

Scikit-learn: For the machine learning pipeline (RandomForestRegressor, StandardScaler).

Joblib: For loading and saving the trained model.

Numpy: For numerical operations.

Matplotlib: For creating the "Feature Relationships" graph.
