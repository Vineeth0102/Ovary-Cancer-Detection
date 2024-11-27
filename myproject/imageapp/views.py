import os
from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .forms import RegistrationForm, LoginForm, ImageUploadForm
from .models import CustomUser

# Load your model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'New_Test_model.keras')
model = load_model(MODEL_PATH)

# Define the mapping for model output
CLASS_MAPPING = {
    0: 'Clear_Cell',
    1: 'Endometri',
    2: 'Mucinous',
    3: 'Non_Cancerous',
    4: 'Serous'
}

def preprocess_image(image, target_size):
    """Preprocess the image to match the input format of the model."""
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = RegistrationForm()
    return render(request, 'imageapp/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'imageapp/login.html', {'form': form})

@login_required
def home(request):
    return render(request, 'imageapp/home.html')

@login_required
def image_upload(request):
    context = {}
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the image with the user
            image_instance = form.save(commit=False)
            image_instance.user = request.user
            image_instance.save()

            # Handle the uploaded image
            image_file = form.cleaned_data['image']
            image = Image.open(image_file)

            # Preprocess and predict
            processed_image = preprocess_image(image, target_size=(224, 224))
            predictions = model.predict(processed_image).flatten()

            highest_class_index = np.argmax(predictions)
            highest_class_name = CLASS_MAPPING[highest_class_index]
            highest_probability = predictions[highest_class_index]

            context['predictions'] = {CLASS_MAPPING[i]: prob for i, prob in enumerate(predictions)}
            context['highest_class'] = highest_class_name
            context['highest_probability'] = highest_probability
    else:
        form = ImageUploadForm()

    context['form'] = form
    return render(request, 'imageapp/upload.html', context)
