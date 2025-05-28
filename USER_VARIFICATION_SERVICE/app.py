# import json, os
# from src.answer_evaluation import *
# from src.face_monitoring_inference import *
# from flask import Flask, request, Response
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# app = Flask(__name__)
# app.config['UPLOAD_IMAGE_FOLDER'] = 'store/images'
# app.config['UPLOAD_AUDIO_FOLDER'] = 'store/audios'
# app.config['UPLOAD_CV_FOLDER'] = 'store/cvs'
# CORS(app)

# # Ensure upload folders exist
# for folder in [app.config['UPLOAD_IMAGE_FOLDER'], app.config['UPLOAD_AUDIO_FOLDER'], app.config['UPLOAD_CV_FOLDER']]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# # @app.route('/api/face_detection', methods=['POST'])
# # def api_face_detection():
# #     try:
# #         username = request.form['username']
# #         image_file = request.files['image_file']

# #         save_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], secure_filename(image_file.filename))
# #         image_file.save(save_path)

# #         head_pose_text, det_username = face_image_inference(
# #             username,
# #             save_path
# #         )

# #         print("user name : "+username)

# #         return Response(
# #             response=json.dumps({
# #                 "Head Pose": head_pose_text,
# #                 "Username": det_username
# #             }),
# #             status=200,
# #             mimetype="application/json"
# #         )

# #     except Exception as e:
# #         return Response(
# #             response=json.dumps({
# #                 "message": "Face detection failed",
# #                 "error": str(e)
# #             }),
# #             status=400,
# #             mimetype="application/json"
# #         )

# @app.route('/api/face_detection', methods=['POST'])
# def api_face_detection():
#     try:
#         username = request.form['username']
#         image_file = request.files['image_file']

#         save_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], secure_filename(image_file.filename))
#         image_file.save(save_path)
        
#         print(f"--- [DEBUG] Username: {username}") # DEBUG
#         print(f"--- [DEBUG] Image saved to: {save_path}") # DEBUG

#         head_pose_text, det_username = face_image_inference(
#             username,
#             save_path
#         )
        
#         print(f"--- [DEBUG] face_image_inference returned: head_pose='{head_pose_text}', detected_user='{det_username}'") # DEBUG

#         return Response(
#             response=json.dumps({
#                 "Head Pose": head_pose_text,
#                 "Username": det_username
#             }),
#             status=200,
#             mimetype="application/json"
#         )

#     except Exception as e:
#         # --- IMPORTANT: Print the full exception details ---
#         import traceback
#         print("--- [ERROR] Exception in /api/face_detection: ---")
#         print(f"--- [ERROR] Exception Type: {type(e)}")
#         print(f"--- [ERROR] Exception Message: {str(e)}")
#         print("--- [ERROR] Traceback: ---")
#         traceback.print_exc() # This will print the full stack trace
#         print("-------------------------------------------------")
#         # --- End of important print ---

#         return Response(
#             response=json.dumps({
#                 "message": "Face detection failed",
#                 "error": str(e) # This might be empty if str(e) is empty
#             }),
#             status=400,
#             mimetype="application/json"
#         )

# @app.route('/api/face_monitoring', methods=['POST'])
# def api_face_monitoring():
#     try:
#         username = request.form['username']
#         response = face_analysis(username)
#         return Response(
#             response=json.dumps(response),
#             status=200,
#             mimetype="application/json"
#         )

#     except Exception as e:
#         return Response(
#             response=json.dumps({
#                 "message": "Face monitoring failed",
#                 "error": str(e)
#             }),
#             status=400,
#             mimetype="application/json"
#         )

   

# @app.route('/api/answer_evaluation', methods=['POST'])
# def api_answer_evaluation():
#     try:
#         question = request.form['question']
#         correct_answer = request.form['correct_answer']
#         user_answer = request.form['user_answer']

#         response = inference_answer_evaluation(question, correct_answer, user_answer)

#         return Response(
#             response=json.dumps({
#                 "Score": response
#             }),
#             status=200,
#             mimetype="application/json"
#         )

#     except Exception as e:
#         return Response(
#             response=json.dumps({
#                 "message": "Answer evaluation failed",
#                 "error": str(e)
#             }),
#             status=400,
#             mimetype="application/json"
#         )


# if __name__ == '__main__':
#     app.run(
#         debug=True,
#         host='0.0.0.0',
#         port=5002
#     )



import json
import os
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS
import traceback # For detailed error logging

# Import your inference module
# This will also trigger the FAISS initialization if it's at the bottom of inference.py
from src.face_monitoring_inference import (
    face_image_inference,
    face_analysis,
    video_face_inference, # If you expose this via API
    initialize_faiss_system # Import this to call it explicitly if needed
)

# --- Initialize FAISS System when Flask app starts ---
# This is crucial. Call it once.
# Set force_build to True only if you want to rebuild the index on every app start (not recommended for production)
print("--- Initializing FAISS System from app.py ---")
initialize_faiss_system(force_build=False)
# ----------------------------------------------------


app = Flask(__name__)
app.config['UPLOAD_IMAGE_FOLDER'] = 'store/images'
# app.config['UPLOAD_AUDIO_FOLDER'] = 'store/audios' # Not used in provided snippets
# app.config['UPLOAD_CV_FOLDER'] = 'store/cvs' # Not used in provided snippets
CORS(app)

# Ensure upload folders exist
for folder_key in ['UPLOAD_IMAGE_FOLDER']: # Add other folder keys if used
    folder_path = app.config[folder_key]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")


@app.route('/api/face_detection', methods=['POST'])
def api_face_detection():
    try:
        if 'username' not in request.form:
            return Response(response=json.dumps({"message": "Username is required"}), status=400, mimetype="application/json")
        if 'image_file' not in request.files:
            return Response(response=json.dumps({"message": "Image file is required"}), status=400, mimetype="application/json")

        username = request.form['username']
        image_file = request.files['image_file']
        
        if image_file.filename == '':
            return Response(response=json.dumps({"message": "No selected image file"}), status=400, mimetype="application/json")

        filename = secure_filename(image_file.filename)
        save_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], filename)
        image_file.save(save_path)
        
        print(f"--- [API DEBUG] User: {username}, Image saved: {save_path}")

        # Call the inference function
        # face_image_inference now uses the globally loaded FAISS index
        head_pose_text, det_username = face_image_inference(
            username, # This is the expected username
            save_path
        )
        
        print(f"--- [API DEBUG] Inference result: HeadPose='{head_pose_text}', DetectedUser='{det_username}'")

        return Response(
            response=json.dumps({
                "Head Pose": head_pose_text,
                "Username": det_username # This is the recognized username
            }),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        print("--- [API ERROR] Exception in /api/face_detection: ---")
        print(f"--- [API ERROR] Type: {type(e)}")
        print(f"--- [API ERROR] Msg: {str(e)}")
        print("--- [API ERROR] Traceback: ---")
        traceback.print_exc()
        print("-------------------------------------------------")
        return Response(
            response=json.dumps({
                "message": "Face detection process failed on server.",
                "error_details": str(e) # Provide a bit more detail if possible
            }),
            status=500, # Internal Server Error for unhandled exceptions
            mimetype="application/json"
        )

@app.route('/api/face_monitoring', methods=['POST'])
def api_face_monitoring():
    try:
        if 'username' not in request.form:
            return Response(response=json.dumps({"message": "Username is required"}), status=400, mimetype="application/json")
        
        username = request.form['username']
        response_data = face_analysis(username) # face_analysis now returns a dict

        if "error" in response_data: # Check if face_analysis returned an error
            return Response(response=json.dumps(response_data), status=400, mimetype="application/json")

        return Response(
            response=json.dumps(response_data),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        print(f"--- [API ERROR] Exception in /api/face_monitoring: {str(e)} ---")
        traceback.print_exc()
        return Response(
            response=json.dumps({
                "message": "Face monitoring analysis failed on server.",
                "error_details": str(e)
            }),
            status=500,
            mimetype="application/json"
        )

# Your answer_evaluation endpoint (unchanged from your original)
# @app.route('/api/answer_evaluation', methods=['POST'])
# def api_answer_evaluation():
    # try:
        # question = request.form['question']
        # correct_answer = request.form['correct_answer']
        # user_answer = request.form['user_answer']
        # Assuming inference_answer_evaluation is defined or imported
        # response = inference_answer_evaluation(question, correct_answer, user_answer)
        # return Response(response=json.dumps({"Score": response}), status=200, mimetype="application/json")
    # except Exception as e:
        # return Response(response=json.dumps({"message": "Answer eval failed", "error": str(e)}), status=400, mimetype="application/json")


if __name__ == '__main__':
    # The FAISS system should be initialized before app.run()
    # It's already called when src.face_monitoring_inference is imported,
    # but an explicit call here (if not done at import) would also work.
    # initialize_faiss_system(force_build=False) # Ensure it's called if not at import.
    
    print("Starting Flask application...")
    app.run(
        debug=True, # Set to False in production
        host='0.0.0.0',
        port=5002
    )