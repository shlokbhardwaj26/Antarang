from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import random

# Define emotion labels
emotion_mapping = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Happy',
    3: 'Fear',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

teenager_angry = [
    "It seems you are feeling angry! Take a few deep breaths to calm down. Deep breathing is a simple yet effective technique to help regulate emotions, especially anger. By taking slow, deep breaths, teenagers can activate their body's relaxation response, which can help reduce feelings of anger and tension.",
    "It seems you are feeling angry! Anger is natural, but it's important to express it constructively. Maybe take a break or go for a walk. Acknowledging the validity of their anger while encouraging constructive expression is essential. Suggesting a break or physical activity like walking allows teenagers to step away from the situation, cool down, and regain perspective before addressing the issue.",
    "It seems you are feeling angry! Try journaling or talking to someone you trust about what's bothering you. Journaling provides a private outlet for teenagers to express their feelings and process their anger. Alternatively, talking to a trusted friend, family member, or mentor can offer support, perspective, and possibly solutions to the source of their anger.",
    "It seems you are feeling angry! Practice mindfulness techniques to manage your anger, like meditation or progressive muscle relaxation. Mindfulness techniques help teenagers become more aware of their thoughts, emotions, and bodily sensations. Meditation and progressive muscle relaxation are proven methods to reduce stress and anger by promoting relaxation and emotional regulation.",
    "It seems you are feeling angry! If you're feeling overwhelmed, consider reaching out to a counselor or therapist for support. Encouraging teenagers to seek professional help when their anger becomes overwhelming emphasizes the importance of mental health support. Counselors or therapists can provide coping strategies, validate their emotions, and help address underlying issues contributing to their anger."
]

teenager_happy = [
    "You look happy! Keep doing things that make you smile. If you're feeling happy, keep doing things that bring you joy and make you smile.",
    "You look happy! Share your happiness with others. Sharing your happiness with friends or family can spread joy and strengthen your relationships.",
    "You look happy! Do things that make you happy, like hanging out with friends. Spending time with friends or doing activities you enjoy can help maintain your happiness and positive mood.",
    "You look happy! Be thankful for good things in your life. Practicing gratitude for the positive aspects of your life can enhance feelings of happiness and contentment.",
    "You look happy! Enjoy feeling happy and remember it when you're sad. Embrace moments of happiness and hold onto them as reminders of joy during difficult times."
]

teenager_sad = [
    "Feeling down? It's okay to be sad. Talk to a friend. If you're feeling sad, it's okay to talk about it with someone you trust, like a friend. They can listen and support you.",
    "Feeling down? Do things that make you happy, like listening to music. Doing things that bring you joy, like listening to music, can help lift your spirits when you're feeling sad.",
    "Feeling down? Take care of yourself by doing nice things. Doing things that make you feel good, like taking a warm bath or treating yourself, can help you feel better when you're sad.",
    "Feeling down? Draw or write about your feelings. Expressing your feelings through art or writing can help you process your emotions and feel better when you're sad.",
    "Feeling down? If you're sad a lot, talk to a counselor. If you're feeling sad often and it's hard to cope, talking to a counselor can help. They can give you support and strategies to feel better."
]

teenager_disgusted = [
    "You look disgusted! If something grosses you out, it's okay to step away from it. Take a break. When you feel disgusted by something, it's okay to move away from it for a while. Give yourself a break from it.",
    "You look disgusted! Talk to someone about why you feel disgusted. It can help you feel better. Sharing your feelings of disgust with someone you trust can make you feel less upset. Talking about it can help you understand why you feel that way.",
    "You look disgusted! Try to focus on things that don't make you feel disgusted. It can help you feel better. Instead of thinking about what disgusts you, try to focus on things that make you feel good. It can help shift your attention away from the unpleasant feeling.",
    "You look disgusted! Take deep breaths and try to calm down if you're feeling overwhelmed by disgust. If you're feeling really disgusted and it's hard to handle, try taking deep breaths to calm yourself down. It can help you feel more in control of your emotions.",
    "You look disgusted! Remember that it's okay to feel disgusted sometimes. Take care of yourself and do things that make you feel better. Feeling disgusted is a normal emotion, and it's okay to experience it. Take care of yourself by doing things that make you feel happier and more comfortable."
]

teenager_fearful = [
    "Feeling scared? It's okay to talk about it with someone you trust. They can help you feel better. When you're feeling scared, it's important to share your feelings with someone you trust, like a friend or family member. They can offer support and comfort.",
    "Feeling scared? Try to understand what's making you feel scared. Sometimes knowing why can make it less scary. Understanding what's causing your fear can help you feel more in control. It's like shining a light on the dark to see that there's nothing to be afraid of.",
    "Feeling scared? Do things that make you feel safe and calm. It can help you feel less scared. Engaging in activities that make you feel secure and relaxed, like listening to calming music or spending time with loved ones, can help ease feelings of fear.",
    "Feeling scared? Take deep breaths to help calm yourself down if you're feeling really scared. When you're feeling overwhelmed by fear, taking deep breaths can help calm your mind and body. It's like pressing a reset button on your emotions.",
    "Feeling scared? Remember that it's okay to feel scared sometimes. Take things one step at a time. Feeling scared is a natural emotion, and it's okay to experience it. Take things one step at a time and be kind to yourself as you navigate your feelings."
]

teenager_surprised = [
    "That's surprising! See where it takes you. When something unexpected happens, embrace it and see what opportunities or new experiences it brings.",
    "You look surprised! Think about how you feel and what you can do next. Take a moment to understand your reaction to the surprise and consider your options for how to respond.",
    "You look surprised! Surprises can be fun! Enjoy the moment. Surprises can bring excitement and joy, so embrace the unexpected and enjoy the moment.",
    "You look surprised! Life is full of surprises. Embrace them! Life can be unpredictable, so be open to surprises and see them as opportunities for growth and adventure.",
    "You look surprised! If you're overwhelmed, take a deep breath and stay calm. If the surprise is too much to handle, take a moment to breathe deeply and stay calm so you can think clearly about how to respond."
]

teenager_neutral = [
    "You seem neutral. Take a moment to think about how you feel. If you're feeling neutral, pause for a moment to check in with yourself and understand your emotions.",
    "You seem neutral! Try something new to make your day more interesting. Exploring new activities or hobbies can add excitement and variety to your day when you're feeling neutral.",
    "You seem neutral! It's okay to feel neutral. Relax and take care of yourself. Feeling neutral is normal, so take this time to relax and focus on self-care to maintain your well-being.",
    "You seem neutral! Take a break from busy things if you need to. If you're feeling overwhelmed, it's okay to take a break from your responsibilities and give yourself some time to recharge.",
    "You seem neutral! Set small goals to feel more motivated. Setting achievable goals can provide a sense of direction and motivation, even when you're feeling neutral."

]

model_path = r"C:\Users\vrind\Documents\project\Flask\REAL_MODEL.h5"

model = load_model(model_path)
img_size = 48

def predict_emotion(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(img_size, img_size), color_mode = "grayscale")
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.reshape(1,48,48,1)
    
    # Load the model
    model = load_model(model_path)
    
    # Predict the emotion
    prediction = model.predict(img)
    
    # Map prediction to emotion label
    predicted_emotion = emotion_mapping[np.argmax(prediction)]
    
    return predicted_emotion



from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from . import db
import base64
import os

views = Blueprint('views', __name__)

@views.route('/', methods=['GET','POST'])
@login_required
def home():
    return render_template("home.html")

@views.route('/about-us')
def aboutUs():
    return render_template("about_us.html")

@views.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the JSON data from the request
        data = request.json

        # Extract the image data from the JSON
        image_data_url = data['image']

        # Extract the base64 encoded image data
        _, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)

        # Define the file path to save the image
        image_path = os.path.join(views.root_path, 'uploads', 'captured_image.jpeg')
        

        # Write the image data to a file
        with open(image_path, 'wb') as f:
            f.write(image_data)

        def out_put(predicted_emotion):
             if predicted_emotion == "Angry" :
                 response = random.choice(teenager_angry)

             elif predicted_emotion == "Sad" :
                 response = random.choice(teenager_sad)
                

             elif predicted_emotion == "Happy" :
                 response = random.choice(teenager_happy)
                

                
             elif predicted_emotion == "Neutral" :
                 response = random.choice(teenager_neutral)
                

                
             elif predicted_emotion == "Fear" :
                 response = random.choice(teenager_fearful)
                 
             elif predicted_emotion == "Surprise" :
                 response = random.choice(teenager_surprised)
                
             else:
                 response = random.choice(teenager_disgusted)
                
             return(response)
        

        predicted_emotion = predict_emotion(image_path)
        # print("Predicted Emotion:", predicted_emotion)
        response=out_put(predicted_emotion)
        # Return a response
        return {'message': response}, 200
    except Exception as e:
        # Handle exceptions
        return {'error': str(e)}, 500

