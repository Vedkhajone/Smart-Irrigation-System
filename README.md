# 🌿 Smart Irrigation System Using Machine Learning 🌿
An intelligent irrigation system that uses real-time environmental data and a machine learning model to predict optimal irrigation needs. Designed to integrate sensor data, Raspberry Pi, and predictive analytics to automate water management and enhance agricultural efficiency.

## 📂 Project Structure
```bash
Smart-Irrigation-System/
├── dataset/                # Collected CSV data files
├── raspberry_pi_code/
│   ├── train_model.py      # Model training script
│   ├── predict.py          # Prediction script (for Raspberry Pi)
│   └── requirements.txt    # Python dependencies
├── models/
│   └── trained_model.pkl   # Saved machine learning model
├── README.md               # Project documentation
└── LICENSE                 # Open-source license 
```
##🚀 Features
- ✅ Real-time data collection from sensors (e.g., temperature, humidity, wind, etc.)
- ✅ Machine learning-based prediction of irrigation needs
- ✅ CSV data logging and automatic training
- ✅ Runs on Raspberry Pi for field deployment

##📦 Requirements
- Python 3.8+
- Pandas, Scikit-learn
- Raspberry Pi (for deployment)
- Sensor Modules (e.g., DHT11, soil moisture, wind sensor)
- 
## Install dependencies:
```bash
pip install -r raspberry_pi_code/requirements.txt
```
## 📚 Data Collection
- 1️⃣ Connect sensors to Raspberry Pi.
- 2️⃣ Periodically log sensor data into CSV files under the dataset/ directory.
- 3️⃣ Ensure consistent column names:

```
id, DateTime, max_temp, min_temp, avg_temp, rh_max, rh_min, rh_avg, L, et0, dew_max, dew_min, dew_avg, Wind_maxMS

```
## 🏋️‍♂️ Model Training
Run the following script to train the model:
```bash
cd raspberry_pi_code
python train_model.py
```
The script reads all CSV files from dataset/, preprocesses them, and trains a machine learning model (e.g., Random Forest).
The trained model is saved as models/trained_model.pkl.

## 🔮 Prediction on Raspberry Pi
Once the model is trained, use the predict.py script to make real-time predictions based on new sensor data:

```bash
cd raspberry_pi_code
python predict.py
```
This script reads the latest data (CSV or live input), loads the trained model, and predicts the required irrigation amount.

## 📈 Example Workflow
- 1️⃣ Collect Data: Sensors on Raspberry Pi log data into CSV files.
- 2️⃣ Train Model: Transfer data to the dataset/ folder and run train_model.py.
- 3️⃣ Deploy Model: Copy trained_model.pkl to Raspberry Pi.
- 4️⃣ Predict: Run predict.py on Raspberry Pi with live sensor data.

## 📝 To-Do
 Integrate live sensor data input into predict.py
 Optimize feature selection
 Implement a simple web dashboard (Flask/Streamlit) for control and monitoring

## 📄 License
This project is licensed under the MIT License. See LICENSE for details.

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgments
Built with 🤍 by Ved Khajone
