from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = r"D:\main\alzheimers_prediction\alzheimer_prediction_model.h5"
model = load_model(MODEL_PATH)

# Categories
CATEGORIES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
IMG_SIZE = 128

# Convert grayscale to RGB
def convert_to_rgb(image):
    return np.repeat(image, 3, axis=-1)

# Detailed descriptions for each class
DEMENTIA_INFO = {
    "Non Demented": {
        "description": "This refers to individuals with no significant cognitive decline. They function independently in daily life, with intact memory, language, and executive function. Minor forgetfulness related to aging may occur but does not impair daily activities. Neuropsychological tests typically fall within normal ranges. Regular assessments help monitor for early signs of cognitive change.",
        "symptoms": [
            "No memory loss affecting daily life",
            "Normal reasoning and language skills",
            "Minor forgetfulness (e.g., misplacing items)",
            "No interference with social or occupational function",
            "Stable behavior and cognitive performance"
        ],
        "diagnosis": [
            "Normal performance on cognitive tests relative to age and education",
            "No functional impairments in daily living",
            "Absence of noticeable cognitive or behavioral decline",
            "Informant reports and clinical interviews confirm intact cognition"
        ]
    },
    "Very mild Dementia": {
        "description": "Characterized by subtle memory lapses, especially with recent events or names, though they may go unnoticed by others. Individuals remain largely independent but may begin using memory aids or require slightly more effort in complex tasks. Changes are often detected through close observation or cognitive screening tools. It may correspond to the earliest stage of Alzheimer’s or mild cognitive impairment (MCI).",
        "symptoms": [
            "Occasional memory lapses (e.g., forgetting recent events or conversations)",
            "Slight difficulty with complex tasks (e.g., planning or multitasking)",
            "Subtle word-finding issues",
            "Still able to function independently",
            "Increased effort or time needed for routine mental tasks"
        ],
        "diagnosis": [
            "Mild cognitive changes noticeable to the person or close contacts",
            "Objective evidence of decline on cognitive assessments",
            "No significant interference with work or social activities",
            "Often categorized as Mild Cognitive Impairment (MCI)"
        ]
    },
    "Mild Dementia": {
        "description": "Symptoms become more apparent and begin to interfere with daily activities. Individuals may have trouble remembering recent conversations, managing finances, or organizing tasks. Personality changes and disorientation may appear. A thorough medical history, cognitive assessment, and imaging studies are often used to establish a diagnosis.",
        "symptoms": [
            "Noticeable short-term memory loss",
            "Difficulty with complex tasks (e.g., managing bills, appointments)",
            "Word-finding difficulty and repetitive questioning",
            "Mild disorientation in time or unfamiliar places",
            "Subtle personality or mood changes"
        ],
        "diagnosis": [
            "Impairment in at least one cognitive domain (e.g., memory, language, executive function)",
            "Decline from previous level of function",
            "Interference with independence in daily activities",
            "Clinical diagnosis based on interviews, cognitive testing, and reports from family or caregivers"
        ]
    },
    "Moderate Dementia": {
        "description": "At this stage, cognitive decline is significant. Patients struggle with daily tasks like dressing, meal prep, or remembering personal history. Language, reasoning, and spatial skills are affected, and behavioral changes such as agitation or withdrawal may emerge. Diagnosis requires detailed clinical evaluation, caregiver reports, and neuroimaging to assess brain changes.",
        "symptoms": [
            "Marked memory loss (e.g., forgetting names of close family or personal history)",
            "Trouble with basic daily tasks (e.g., dressing, hygiene, cooking)",
            "Language becomes impaired, often with trouble forming coherent speech",
            "Confusion, poor judgment, or risk of wandering",
            "Behavioral changes such as aggression, withdrawal, or depression"
        ],
        "diagnosis": [
            "Deficits in multiple cognitive domains (memory, language, visuospatial, etc.)",
            "Clear and progressive functional impairment",
            "Dependent on others for daily activities",
            "Confirmed through clinical evaluation, caregiver reports, and often brain imaging (e.g., MRI, CT)"
        ]
    }
}

# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        img_rgb = convert_to_rgb(img)

        # Predict
        pred = model.predict(img_rgb)
        class_idx = np.argmax(pred, axis=1)[0]
        prediction = CATEGORIES[class_idx]
        info = DEMENTIA_INFO[prediction]

        # Prepare image for result display
        image_url = url_for('static', filename=f'uploads/{filename}')
        static_upload_path = os.path.join('static/uploads', filename)
        os.makedirs(os.path.dirname(static_upload_path), exist_ok=True)
        cv2.imwrite(static_upload_path, cv2.imread(file_path))

        return render_template('result.html',
                               prediction=prediction,
                               description=info["description"],
                               symptoms=info["symptoms"],
                               diagnosis=info["diagnosis"],
                               image_url=image_url)

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
