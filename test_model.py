import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Sample input data (replace with actual test values)
sample_data = {
    'Nitrogen': 20,
    'Phosporus': 30,
    'Potassium': 40,
    'Temperature': 25.5,
    'Humidity': 60.0,
    'Ph': 6.5,
    'Rainfall': 100.0
}

# Convert input data to array
feature_list = [sample_data['Nitrogen'], sample_data['Phosporus'], sample_data['Potassium'],
                sample_data['Temperature'], sample_data['Humidity'], sample_data['Ph'],
                sample_data['Rainfall']]
single_pred = np.array(feature_list).reshape(1, -1)

# Apply scaling
scaled_features = ms.transform(single_pred)
final_features = sc.transform(scaled_features)

# Make prediction
prediction = model.predict(final_features)

# Map prediction to crop
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# Print result
if prediction[0] in crop_dict:
    crop = crop_dict[prediction[0]]
    print(f"Recommended Crop: {crop}")
else:
    print("Sorry, we could not determine the best crop to be cultivated with the provided data.")
