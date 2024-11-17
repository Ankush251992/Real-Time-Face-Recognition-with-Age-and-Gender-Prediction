import os
import cv2
import numpy as np
import insightface
from numpy.linalg import norm


# Function to compute cosine similarity between two embeddings
def cosine_similarity(emb1, emb2):
    return emb1.dot(emb2) / (norm(emb1) * norm(emb2))


# Load images from the employee_images folder, detect faces, and extract embeddings
def load_embeddings_from_images(folder, recognizer):
    face_database = {}
    for file in os.listdir(folder):
        if file.endswith((".jpg", ".jpeg", ".png")):  # Only load image files
            # Get the name from the filename (without extension)
            name = os.path.splitext(file)[0]
            img_path = os.path.join(folder, file)

            # Read the image
            img = cv2.imread(img_path)

            # Detect faces in the image
            faces = recognizer.get(img)

            # Ensure there is only one face detected per image
            if len(faces) == 1:
                # Extract embedding for the detected face
                embedding = faces[0].embedding
                face_database[name] = embedding
                print(f"Added {name} to the face database.")
            else:
                print(f"Skipping {name}, as it contains none or multiple faces.")

    return face_database


# Initialize face detector and recognition model
def initialize_models():
    # Initialize FaceAnalysis, which uses default models for face detection and recognition
    recognizer = insightface.app.FaceAnalysis()

    # Removed the 'nms' argument as it's no longer needed
    recognizer.prepare(ctx_id=0)  # Use GPU (ctx_id=0) or CPU (ctx_id=-1)

    # The detector is part of the FaceAnalysis model by default
    detector = recognizer.models['detection']

    return detector, recognizer


# Main function for real-time face recognition using webcam
def face_recognition_from_webcam(folder, threshold=0.5):
    # Initialize the models
    detector, recognizer = initialize_models()

    # Load the embeddings from the images folder
    face_database = load_embeddings_from_images(folder, recognizer)
    print(f"Loaded {len(face_database)} known faces from database.")

    # Start webcam feed
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect faces in the frame
        faces = recognizer.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)

            # Extract embedding for the detected face
            embedding = face.embedding

            # Initialize best match variables
            best_match = None
            highest_similarity = 0

            # Compare the detected face's embedding with each entry in the database
            for name, db_embedding in face_database.items():
                sim_score = cosine_similarity(embedding, db_embedding)
                if sim_score > highest_similarity:
                    highest_similarity = sim_score
                    best_match = name

            # Draw a bounding box and label for the recognized face
            if highest_similarity > threshold:
                label = f'{best_match} ({highest_similarity:.2f})'
            else:
                label = 'Unknown'

            # Draw the bounding box around the face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Draw the name label above the face
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Folder where the employee images are stored
    folder = 'employee_images'

    # Run face recognition from webcam feed
    face_recognition_from_webcam(folder)