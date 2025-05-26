# import cv2 as cv
# import cv2, time
# import numpy as np
# import pandas as pd
# import yaml, pymongo
# import mediapipe as mp
# import faiss, glob, os
# from deepface import DeepFace
# from datetime import datetime, timedelta
# import os

# with open('secrets.yaml') as f:
#     secrets = yaml.load(f, Loader=yaml.FullLoader)

# os.environ["MONGO_DB_URI"] = secrets['MONGO_DB_URI']

# try:
#     client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
#     db = client['Elearning']
#     ffeatures_collection = db['ffeatures']
#     print("Connected to MongoDB")
    
# except Exception as e:
#     print(e)

# face_mesh = mp.solutions.face_mesh.FaceMesh(
#                                             min_detection_confidence=0.5, 
#                                             min_tracking_confidence=0.5
#                                             )

# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(
#                                     color=(128,0,128),
#                                     circle_radius=1,
#                                     thickness=2
#                                     )
# p_face_mesh = mp.solutions.face_mesh

# models = [
#         "VGG-Face", 
#         "Facenet", 
#         "Facenet512", 
#         "OpenFace", 
#         "DeepFace", 
#         "DeepID", 
#         "ArcFace", 
#         "Dlib", 
#         "SFace",
#         "GhostFaceNet",
#         ]

# def head_pose_inference(
#                         image,
#                         image_flag = False
#                         ):
#     start = time.time()

#     if image_flag:
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     else:
#         image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) 
#     image.flags.writeable = False

#     results = face_mesh.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

#     img_h , img_w, img_c = image.shape
#     face_2d = []
#     face_3d = []

#     texts = []
#     face_centroids = []
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for idx, lm in enumerate(face_landmarks.landmark):
#                 if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
#                     if idx ==1:
#                         nose_2d = (lm.x * img_w,lm.y * img_h)
#                         nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
#                     x,y = int(lm.x * img_w),int(lm.y * img_h)

#                     face_2d.append([x,y])
#                     face_3d.append(([x,y,lm.z]))

#             face_2d = np.array(face_2d,dtype=np.float64)
#             face_3d = np.array(face_3d,dtype=np.float64)

#             face_centroid = np.mean(face_2d,axis=0)

#             focal_length = 1 * img_w
#             cam_matrix = np.array([[focal_length,0,img_h/2],
#                                   [0,focal_length,img_w/2],
#                                   [0,0,1]])
#             distortion_matrix = np.zeros((4,1),dtype=np.float64)
#             success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)

#             rmat,jac = cv2.Rodrigues(rotation_vec)
#             angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

#             x = angles[0] * 360
#             y = angles[1] * 360
#             z = angles[2] * 360

#             if y < -10:
#                 text="Looking Left"
#             elif y > 10:
#                 text="Looking Right"
#             elif x < -10:
#                 text="Looking Down"
#             elif x > 10:
#                 text="Looking Up"
#             else:
#                 text="Forward"
#             texts.append(text)
#             face_centroids.append(face_centroid)
#             nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

#             p1 = (int(nose_2d[0]),int(nose_2d[1]))
#             p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

#             cv2.line(image,p1,p2,(255,0,0),3)

#             cv2.putText(image,text,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),3)
#             cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#             cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#             cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#         end = time.time()
#         totalTime = end-start

#         fps = 1/totalTime
#         print("FPS: ",fps)

#         cv2.putText(image,f'FPS: {int(fps)}',(20,450),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

#         mp_drawing.draw_landmarks(
#                                 image=image,
#                                 landmark_list=face_landmarks,
#                                 connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#                                 connection_drawing_spec=drawing_spec,
#                                 landmark_drawing_spec=drawing_spec
#                                 )
#     return image, texts, face_centroids

# def extract_face_information_for_db(img_path):
#     face_objs = DeepFace.represent(
#                                 img_path = img_path,
#                                 model_name = models[2],
#                                 enforce_detection = False
#                                 )
#     img_path = img_path.replace("\\", "/")
#     user_name = img_path.split("/")[-2]

#     if len(face_objs) != 1:
#         if len(face_objs) == 0:
#             Warning(f"No faces detected in the image : {img_path}")
#         else:
#             Warning(f"Multiple faces detected in the image : {img_path}")
#         return None, None, None, None

#     else:
#         facial_area = face_objs[0]['facial_area']
#         embeddings = face_objs[0]['embedding']
#         face_confidence = face_objs[0]['face_confidence']
#         x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

#     return embeddings, face_confidence, (x, y, w, h), user_name

# def build_face_embedding_index(
#                                 d = 128,
#                                 face_index_path = 'models/face_index',
#                                 face_image_dir = 'data/facedb/*/*.jpg',
#                                 face_details_path = 'models/face_details.npz',
#                                 ):
#     if (not os.path.exists(face_index_path)) or (not os.path.exists(face_details_path)):
#         faiss_index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

#         embeddings = []
#         user_names = []
#         facial_areas = []
#         face_confidences = []
        
#         for idx, img_path in enumerate(glob.glob(face_image_dir)):
#             emb, face_confidence, facial_area, user_name = extract_face_information_for_db(img_path)
#             if emb is not None:
#                 embeddings.append(emb)
#                 user_names.append(user_name)
#                 facial_areas.append(facial_area)
#                 face_confidences.append(face_confidence)

#             if idx % 10 == 0:
#                 print(f"Processed {idx}/{len(glob.glob(face_image_dir))} images")

#         embeddings = np.asarray(embeddings).astype('float32')
#         faiss.normalize_L2(embeddings)
#         faiss_index.add(embeddings)
#         faiss.write_index(faiss_index, face_index_path)

#         np.savez(
#                 face_details_path, 
#                 user_names=user_names, 
#                 facial_areas=facial_areas, 
#                 face_confidences=face_confidences
#                 )
        
#     else:
#         faiss_index = faiss.read_index(face_index_path)
#         face_details = np.load(face_details_path)
#         user_names = face_details['user_names']
#         facial_areas = face_details['facial_areas']
#         face_confidences = face_details['face_confidences']

#     return faiss_index, user_names, facial_areas, face_confidences

# # def extract_face_information_for_inference(img_path):
# #     face_objs = DeepFace.represent(
# #                                 img_path = img_path,
# #                                 model_name = models[2],
# #                                 enforce_detection = False
# #                                 )
# #     img_path = img_path.replace("\\", "/")

# #     embeddings = []
# #     facial_areas = []
# #     face_confidences = []

# #     if len(face_objs) == 0:
# #         Warning(f"No faces detected in the image : {img_path}")
# #     else:
# #         for i in range(len(face_objs)):
# #             embs = face_objs[i]['embedding']
# #             facial_area = face_objs[i]['facial_area']
# #             face_confidence = face_objs[i]['face_confidence']
# #             x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

# #             embeddings.append(embs)
# #             facial_areas.append((x, y, w, h))   
# #             face_confidences.append(face_confidence)

# #     return embeddings, face_confidences, facial_areas


# def extract_face_information_for_inference(img_path):
#     print(f"--- [EXTRACT_INFERENCE_DEBUG] Starting extraction for: {img_path}")
#     try:
#         face_objs = DeepFace.represent(
#                                     img_path = img_path,
#                                     model_name = models[2], # OpenFace
#                                     enforce_detection = False # Try with True also to see if face is detected
#                                     )
#         print(f"--- [EXTRACT_INFERENCE_DEBUG] RAW face_objs from DeepFace.represent: {face_objs}")
#     except Exception as e_repr:
#         print(f"--- [EXTRACT_INFERENCE_ERROR] Exception in DeepFace.represent: {e_repr}")
#         import traceback
#         traceback.print_exc()
#         return [], [], []


#     # img_path = img_path.replace("\\", "/") # Already done if needed

#     embeddings = []
#     facial_areas = []
#     face_confidences = []

#     if not face_objs: # Check if face_objs is empty or None
#         print(f"--- [EXTRACT_INFERENCE_WARN] No face objects returned by DeepFace.represent for {img_path}")
#         # Warning(f"No faces detected in the image : {img_path}") # Your original warning
#     elif isinstance(face_objs, list) and len(face_objs) > 0 and isinstance(face_objs[0], dict) and 'embedding' in face_objs[0]:
#          print(f"--- [EXTRACT_INFERENCE_DEBUG] DeepFace.represent returned {len(face_objs)} face object(s).")
#          for i in range(len(face_objs)):
#             face_obj = face_objs[i]
#             print(f"--- [EXTRACT_INFERENCE_DEBUG] Processing face object {i}: Confidence: {face_obj.get('face_confidence', 'N/A')}")
            
#             # Ensure all expected keys are present before trying to access them
#             if 'embedding' in face_obj and 'facial_area' in face_obj and 'face_confidence' in face_obj:
#                 embs = face_obj['embedding']
#                 facial_area_dict = face_obj['facial_area']
#                 face_confidence = face_obj['face_confidence']
                
#                 # Ensure facial_area_dict has x, y, w, h
#                 if all(k in facial_area_dict for k in ('x', 'y', 'w', 'h')):
#                     x, y, w, h = facial_area_dict['x'], facial_area_dict['y'], facial_area_dict['w'], facial_area_dict['h']
#                     embeddings.append(embs)
#                     facial_areas.append((x, y, w, h))
#                     face_confidences.append(face_confidence)
#                     print(f"--- [EXTRACT_INFERENCE_DEBUG] Appended embedding for face {i}. Face confidence: {face_confidence}")
#                 else:
#                     print(f"--- [EXTRACT_INFERENCE_WARN] Facial area keys missing for face object {i}")
#             else:
#                 print(f"--- [EXTRACT_INFERENCE_WARN] Expected keys (embedding, facial_area, face_confidence) missing in face object {i}")
#     else:
#         # This case handles if DeepFace.represent returns something unexpected
#         # (e.g. not a list of dicts, or dicts without 'embedding')
#         print(f"--- [EXTRACT_INFERENCE_ERROR] Unexpected return type or structure from DeepFace.represent: {type(face_objs)}")
#         if isinstance(face_objs, list) and len(face_objs) > 0:
#             print(f"--- [EXTRACT_INFERENCE_ERROR] First element type: {type(face_objs[0])}, Keys: {face_objs[0].keys() if isinstance(face_objs[0], dict) else 'Not a dict'}")


#     print(f"--- [EXTRACT_INFERENCE_DEBUG] Returning {len(embeddings)} embeddings, {len(face_confidences)} confidences, {len(facial_areas)} facial_areas.")
#     return embeddings, face_confidences, facial_areas

# # def search_face_in_db(
# #                     img_path, 
# #                     face_index_path = 'models/face_index',
# #                     face_details_path = 'models/face_details.npz',
# #                     ):
# #     index, user_names, _, _ = build_face_embedding_index(
# #                                                         d= 128
# #                                                         face_index_path = face_index_path,
# #                                                         face_details_path = face_details_path,
# #                                                         )
# #     embeddings, face_confidences, facial_areas = extract_face_information_for_inference(img_path)

# #before adding 1


# # def search_face_in_db(
# #                     img_path,
# #                     face_index_path = 'models/face_index',
# #                     face_details_path = 'models/face_details.npz',
# #                     # d_value = 128 # You can add it as a parameter to search_face_in_db if needed elsewhere
# #                     ):
# #     index, user_names, _, _ = build_face_embedding_index(
# #                                                         d = 128,  # <--- ********  ADD THIS LINE ********
# #                                                         face_index_path = face_index_path,
# #                                                         face_details_path = face_details_path,
# #                                                         )
# #     embeddings, face_confidences, facial_areas = extract_face_information_for_inference(img_path)

# #     retrieved_user_names = []
# #     retrieved_facial_areas = []
# #     retrieved_face_confidences = []

# #     if embeddings is not None:
# #         for idx, emb in enumerate(embeddings):
# #             if face_confidences[idx] >= 0.8:
# #                 emb = np.array(emb).reshape(1, -1).astype('float32')
# #                 faiss.normalize_L2(emb)
# #                 D, I = index.search(emb, 5)
# #                 I = np.array(I).squeeze()
# #                 D = np.array(D).squeeze()
# #                 user_name_list = [user_names[i] for i in I]
# #                 user_name = max(set(user_name_list), key = user_name_list.count)
# #                 avg_confidence = np.mean([d for i, d in zip(I, D) if user_names[i] == user_name])
# #                 retrieved_face_confidences.append(np.round(avg_confidence, 3))
# #                 retrieved_facial_areas.append(facial_areas[idx])
# #                 retrieved_user_names.append(user_name)

# #     return retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences

# #before ading new 2.24 am
# def search_face_in_db(
#                     img_path,
#                     face_index_path = 'models/face_index',
#                     face_details_path = 'models/face_details.npz',
#                     ):
#     print(f"--- [SEARCH_DB_DEBUG] Starting search_face_in_db for: {img_path}")
#     index, user_names_from_index, _, _ = build_face_embedding_index( # Renamed user_names to avoid confusion
#                                                         d = 128,
#                                                         face_index_path = face_index_path,
#                                                         face_details_path = face_details_path,
#                                                         )
#     if index is None:
#         print(f"--- [SEARCH_DB_ERROR] Faiss index is None. Cannot proceed.")
#         return [], [], []
#     if user_names_from_index is None or len(user_names_from_index) == 0:
#         print(f"--- [SEARCH_DB_ERROR] No user names loaded from index. Cannot proceed.")
#         return [], [], []
        
#     print(f"--- [SEARCH_DB_DEBUG] Index loaded. Number of users in index: {len(np.unique(user_names_from_index)) if user_names_from_index is not None else 'None'}. Total entries: {index.ntotal if index else 'None'}")

#     embeddings, face_confidences, facial_areas = extract_face_information_for_inference(img_path)
#     print(f"--- [SEARCH_DB_DEBUG] Extracted from inference: {len(embeddings)} embeddings, confidences: {face_confidences}")

#     retrieved_user_names = []
#     retrieved_facial_areas = []
#     retrieved_face_confidences_for_return = [] # Renamed to avoid confusion

#     if embeddings: # Check if embeddings list is not empty
#         for idx, emb in enumerate(embeddings):
#             print(f"--- [SEARCH_DB_DEBUG] Processing embedding {idx + 1}/{len(embeddings)}. Face confidence from extraction: {face_confidences[idx]}")
#             if face_confidences[idx] >= 0.8: # This is a high threshold for face detection confidence
#                 print(f"--- [SEARCH_DB_DEBUG] Face confidence {face_confidences[idx]} >= 0.8. Proceeding with Faiss search.")
#                 emb_np = np.array(emb).reshape(1, -1).astype('float32')
#                 faiss.normalize_L2(emb_np)
                
#                 k_neighbors = 5 # Number of neighbors to search
#                 print(f"--- [SEARCH_DB_DEBUG] Searching for {k_neighbors} neighbors in Faiss index (ntotal={index.ntotal}).")
                
#                 if index.ntotal == 0:
#                     print(f"--- [SEARCH_DB_WARN] Faiss index is empty (ntotal=0). Skipping search.")
#                     continue # Skip to next embedding if index is empty

#                 D, I = index.search(emb_np, k_neighbors)
#                 print(f"--- [SEARCH_DB_DEBUG] Faiss search results: Distances (D): {D}, Indices (I): {I}")

#                 # Squeeze only if I is not a single row already (e.g. if k_neighbors=1, I might be (1,1))
#                 # I_squeezed = I.squeeze() if I.ndim > 1 and I.shape[0] == 1 else I.flatten() # More robust squeeze
#                 # D_squeezed = D.squeeze() if D.ndim > 1 and D.shape[0] == 1 else D.flatten()
                
#                 # Ensure I and D are 1D arrays for easier processing if k_neighbors > 0
#                 if I.ndim > 1: I = I.flatten()
#                 if D.ndim > 1: D = D.flatten()

#                 # Check if any valid indices were returned (I might contain -1 if fewer than k_neighbors found)
#                 valid_indices = I[I != -1] # Faiss returns -1 for non-existent neighbors
#                 valid_distances = D[I != -1]

#                 if len(valid_indices) == 0:
#                     print(f"--- [SEARCH_DB_WARN] No valid neighbors found in Faiss search for embedding {idx}.")
#                     continue


#                 user_name_list = [user_names_from_index[i_val] for i_val in valid_indices if i_val < len(user_names_from_index)]
#                 print(f"--- [SEARCH_DB_DEBUG] Candidate user names from Faiss: {user_name_list}")

#                 if user_name_list:
#                     user_name = max(set(user_name_list), key = user_name_list.count)
#                     # Get distances for the majority user_name
#                     avg_confidence_list = [d_val for i_val, d_val in zip(valid_indices, valid_distances) if (i_val < len(user_names_from_index) and user_names_from_index[i_val] == user_name)]
                    
#                     if avg_confidence_list:
#                         avg_confidence = np.mean(avg_confidence_list)
#                         print(f"--- [SEARCH_DB_DEBUG] Majority user: {user_name}, Avg. Faiss Match Confidence (Distance): {avg_confidence}")
#                         retrieved_face_confidences_for_return.append(np.round(avg_confidence, 3))
#                         retrieved_facial_areas.append(facial_areas[idx])
#                         retrieved_user_names.append(user_name)
#                     else:
#                         print(f"--- [SEARCH_DB_WARN] No distances found for majority user {user_name}. This shouldn't happen if user_name_list is not empty.")
#                 else:
#                     print(f"--- [SEARCH_DB_WARN] user_name_list is empty after Faiss search for embedding {idx}.")
#             else:
#                 print(f"--- [SEARCH_DB_WARN] Face confidence {face_confidences[idx]} < 0.8. Skipping Faiss search.")
#     else:
#         print(f"--- [SEARCH_DB_WARN] No embeddings extracted. Cannot search in DB.")

#     print(f"--- [SEARCH_DB_DEBUG] search_face_in_db returning: users='{retrieved_user_names}', confidences='{retrieved_face_confidences_for_return}'")
#     return retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences_for_return

# def eculedian_distance(x1, y1, x2, y2):
#     return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# # def face_image_inference(
# #                         username,
# #                         face_image_path
# #                         ):
# #     img = cv2.imread(face_image_path)
# #     img_cp = img.copy()

# #     img_cp, texts, face_centroids = head_pose_inference(img_cp, image_flag = True)

# #     retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db(face_image_path)
# #     for i in range(len(retrieved_user_names)):
# #         x, y, w, h = retrieved_facial_areas[i]
# #         face_centhroid_bbox = (x + w//2, y + h//2)
# #         face_area = w * h
# #         if (face_area >= 20000):
# #             timestamp = datetime.now()
# #             timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
# #             if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
# #                 if (len(face_centroids) > 1) and (len(face_centroids) > 1):
# #                     distances = [eculedian_distance(face_centhroid_bbox[0], face_centhroid_bbox[1], x, y) for x, y in face_centroids]
# #                     head_pose_text = texts[np.argmin(distances)]
# #                 else:
# #                     head_pose_text = texts[0]
                    
# #                 ffeatures_collection.insert_one({
# #                                                 "exp_username": username,
# #                                                 "det_username": retrieved_user_names[i],
# #                                                 "head_pose": head_pose_text,
# #                                                 "face_confidence": float(retrieved_face_confidences[i]),
# #                                                 "timestamp": timestamp
# #                                                 })
# #                 det_username = retrieved_user_names[i]

# #                 cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #                 font = cv.FONT_HERSHEY_SIMPLEX
# #                 cv.putText(img_cp, f'User: {retrieved_user_names[i]}', (x-30, y-40), font, 1, (0, 255, 0), 2)
# #             else:
# #                 cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
# #                 ffeatures_collection.insert_one({
# #                                                 "exp_username": username,
# #                                                 "det_username": "N/A",
# #                                                 "head_pose": "Unknown",
# #                                                 "face_confidence": "N/A",
# #                                                 "timestamp": timestamp
# #                                                 })
# #                 det_username = "N/A"
                
# #     return head_pose_text, det_username
# #     # cv.imshow('Face Monitoring Inference', img_cp)
# #     # cv.waitKey(0)
# #     # cv.destroyAllWindows()

# def face_image_inference(
#                         username,
#                         face_image_path
#                         ):
#     print(f"--- [FACE_INFERENCE_DEBUG] Starting face_image_inference for user: {username}, image: {face_image_path}") # DEBUG START

#     img = cv2.imread(face_image_path)
#     if img is None:
#         print(f"--- [FACE_INFERENCE_ERROR] Could not read image from {face_image_path}")
#         # Decide what to return here if image fails to load
#         # For now, let's raise an error to see it in the main traceback
#         raise ValueError(f"Could not read image: {face_image_path}")
#         # return "Error: Image Load Failed", "N/A" # Alternative: return specific error strings

#     img_cp = img.copy()

#     print(f"--- [FACE_INFERENCE_DEBUG] Calling head_pose_inference...")
#     try:
#         img_cp_posed, texts, face_centroids = head_pose_inference(img_cp, image_flag=True)
#         print(f"--- [FACE_INFERENCE_DEBUG] head_pose_inference returned: texts='{texts}', face_centroids_count={len(face_centroids) if face_centroids else 0}")
#     except Exception as hp_e:
#         print(f"--- [FACE_INFERENCE_ERROR] Exception in head_pose_inference: {hp_e}")
#         import traceback
#         traceback.print_exc()
#         # Decide how to handle head pose error
#         texts = [] # Default to empty if error
#         face_centroids = [] # Default to empty if error
#         # raise # Optionally re-raise to stop processing

#     print(f"--- [FACE_INFERENCE_DEBUG] Calling search_face_in_db...")
#     try:
#         retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db(face_image_path)
#         print(f"--- [FACE_INFERENCE_DEBUG] search_face_in_db returned: users='{retrieved_user_names}', confidences='{retrieved_face_confidences}'")
#     except Exception as sdb_e:
#         print(f"--- [FACE_INFERENCE_ERROR] Exception in search_face_in_db: {sdb_e}")
#         import traceback
#         traceback.print_exc()
#         retrieved_user_names = [] # Default to empty
#         retrieved_facial_areas = []
#         retrieved_face_confidences = []
#         # raise # Optionally re-raise

#     # Initialize return variables with defaults
#     final_head_pose_text = "Unknown"
#     final_det_username = "N/A"

#     if not retrieved_user_names:
#         print(f"--- [FACE_INFERENCE_WARN] No users retrieved from DB for {face_image_path}")
#         # Insert a record for no detection or unknown user if required by your logic
#         # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         # ffeatures_collection.insert_one({
#         #     "exp_username": username,
#         #     "det_username": "N/A_NoDetectionInDB",
#         #     "head_pose": "Unknown_NoDetectionInDB",
#         #     "face_confidence": "N/A",
#         #     "timestamp": timestamp
#         # })

#     for i in range(len(retrieved_user_names)):
#         x, y, w, h = retrieved_facial_areas[i]
#         # face_centhroid_bbox = (x + w//2, y + h//2) # Corrected typo: face_centroid_bbox
#         face_centroid_bbox_x = x + w // 2
#         face_centroid_bbox_y = y + h // 2

#         face_area = w * h
#         print(f"--- [FACE_INFERENCE_DEBUG] Processing retrieved user {i+1}/{len(retrieved_user_names)}: {retrieved_user_names[i]}, area: {face_area}, confidence: {retrieved_face_confidences[i]}")

#         if (face_area >= 20000): # This threshold might be too high for some images
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             current_head_pose_for_db = "Unknown" # Default for this iteration

#             if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
#                 print(f"--- [FACE_INFERENCE_DEBUG] User {username} VERIFIED with confidence {retrieved_face_confidences[i]}")
#                 final_det_username = retrieved_user_names[i] # Set the final detected username

#                 if texts and face_centroids: # Ensure both are available
#                     # Corrected: face_centroid_bbox expects (x,y) not separate x,y in eculedian_distance
#                     distances = [eculedian_distance(face_centroid_bbox_x, face_centroid_bbox_y, fc_x, fc_y) for fc_x, fc_y in face_centroids]
#                     if distances:
#                         min_dist_idx = np.argmin(distances)
#                         if min_dist_idx < len(texts): # Check index bounds for texts
#                             current_head_pose_for_db = texts[min_dist_idx]
#                             final_head_pose_text = current_head_pose_for_db # Update final head pose
#                             print(f"--- [FACE_INFERENCE_DEBUG] Head pose for verified user: {current_head_pose_for_db}")
#                         else:
#                             print(f"--- [FACE_INFERENCE_WARN] Min distance index out of bounds for texts list.")
#                             current_head_pose_for_db = "Error: Text Index OOB" # Out Of Bounds
#                             final_head_pose_text = current_head_pose_for_db

#                     else: # No distances, possibly no face_centroids
#                          print(f"--- [FACE_INFERENCE_WARN] No distances calculated (face_centroids might be empty). Texts: {texts}")
#                          current_head_pose_for_db = texts[0] if texts else "Error: No Centroids/Texts"
#                          final_head_pose_text = current_head_pose_for_db

#                 elif texts: # Only texts available, no face_centroids for matching
#                     print(f"--- [FACE_INFERENCE_WARN] Only texts available, no face_centroids. Using first text element.")
#                     current_head_pose_for_db = texts[0]
#                     final_head_pose_text = current_head_pose_for_db
#                 else: # Neither texts nor face_centroids available
#                     print(f"--- [FACE_INFERENCE_WARN] Neither texts nor face_centroids available for head pose.")
#                     current_head_pose_for_db = "Unknown (No Pose Data)"
#                     final_head_pose_text = current_head_pose_for_db


#                 # DB Insert for verified user
#                 ffeatures_collection.insert_one({
#                                                 "exp_username": username,
#                                                 "det_username": retrieved_user_names[i],
#                                                 "head_pose": current_head_pose_for_db,
#                                                 "face_confidence": float(retrieved_face_confidences[i]),
#                                                 "timestamp": timestamp
#                                                 })
#                 # Drawing logic (optional for debugging server-side)
#                 # cv.rectangle(img_cp_posed, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 # ...

#             else: # User not verified or low confidence
#                 print(f"--- [FACE_INFERENCE_DEBUG] User {retrieved_user_names[i]} NOT verified as {username} OR low confidence ({retrieved_face_confidences[i]})")
#                 # DB Insert for non-verified user (or if it's a different user)
#                 ffeatures_collection.insert_one({
#                                                 "exp_username": username, # Logged in user
#                                                 "det_username": retrieved_user_names[i], # Actual detected user
#                                                 "head_pose": "Unknown (Not Verified)", # Or get from head_pose_inference if available
#                                                 "face_confidence": float(retrieved_face_confidences[i]) if retrieved_face_confidences else "N/A",
#                                                 "timestamp": timestamp
#                                                 })
#                 # final_det_username remains "N/A" or the last verified user if any
#         else:
#             print(f"--- [FACE_INFERENCE_WARN] Face area {face_area} is less than threshold 20000 for user {retrieved_user_names[i]}")

#     print(f"--- [FACE_INFERENCE_DEBUG] Returning: final_head_pose_text='{final_head_pose_text}', final_det_username='{final_det_username}'")
#     return final_head_pose_text, final_det_username

# def video_face_inference(
#                         username,
#                         is_vis = False
#                         ):
#     cap = cv.VideoCapture(0)
#     while cap.isOpened():
#         success, img = cap.read()
#         if not success:
#             break

#         img_cp = img.copy()
#         img_cp_ = cv2.flip(img_cp, 1)
#         cv.imwrite("data/temp_dir/temp.jpg", img_cp_)

#         img_cp, texts, face_centroids = head_pose_inference(img_cp)

#         retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db("data/temp_dir/temp.jpg")
#         for i in range(len(retrieved_user_names)):
#             x, y, w, h = retrieved_facial_areas[i]
#             face_centhroid_bbox = (x + w//2, y + h//2)
#             face_area = w * h
#             if (face_area >= 20000):
#                 timestamp = datetime.now()
#                 timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#                 if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
#                     if (len(face_centroids) > 1) and (len(face_centroids) > 1):
#                         distances = [eculedian_distance(face_centhroid_bbox[0], face_centhroid_bbox[1], x, y) for x, y in face_centroids]
#                         head_pose_text = texts[np.argmin(distances)]
#                     elif len(face_centroids) == 1:
#                         head_pose_text = texts[0]

#                     else:
#                         head_pose_text = "UnKnown"

#                     ffeatures_collection.insert_one({
#                                                     "exp_username": username,
#                                                     "det_username": retrieved_user_names[i],
#                                                     "head_pose": head_pose_text,
#                                                     "face_confidence": float(retrieved_face_confidences[i]),
#                                                     "timestamp": timestamp
#                                                     })
                
#                     cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                     font = cv.FONT_HERSHEY_SIMPLEX
#                     cv.putText(img_cp, f'User: {retrieved_user_names[i]}', (x-30, y-40), font, 1, (0, 255, 0), 2)
#                 else:
#                     cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                     ffeatures_collection.insert_one({
#                                                     "exp_username": username,
#                                                     "det_username": "N/A",
#                                                     "head_pose": "Unknown",
#                                                     "face_confidence": "N/A",
#                                                     "timestamp": timestamp
#                                                     })
                
#         if is_vis:
#             cv.imshow('Face Monitoring Inference', img_cp)
#             if cv.waitKey(5) & 0xFF == 27:
#                 break

#     cap.release()
#     cv.destroyAllWindows()

# def face_analysis(
#                 username,
#                 x_min = 1440
#                 ):
#     current_time = datetime.now()
#     current_time_minus_x = current_time - timedelta(minutes=x_min)

#     current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
#     current_time_minus_x = current_time_minus_x.strftime("%Y-%m-%d %H:%M:%S")

#     data_user = ffeatures_collection.find({
#                                         "exp_username": username,
#                                         "timestamp": {
#                                                     "$gte": current_time_minus_x,
#                                                     "$lt": current_time
#                                                     }
#                                         })
#     data_user = pd.DataFrame(list(data_user))
    
#     if data_user.empty:
#         return None, None
    
#     else:
#         data_user = data_user.drop(columns = ["_id", "exp_username", "face_confidence", "timestamp"])
#         n_forward = len(data_user[data_user["head_pose"] == "Forward"])
#         n_detected = len(data_user[data_user["det_username"] == username])
#         n_total = len(data_user)

#         detected_percentage = (n_detected/n_total)*100
#         forward_percentage = (n_forward/n_total)*100

#         detected_percentage = round(detected_percentage, 2)
#         forward_percentage = round(forward_percentage, 2)


#         detected_percentage = f"{detected_percentage} %"
#         forward_percentage = f"{forward_percentage} %"

#         return {
#                 "detected_percentage": detected_percentage,
#                 "forward_percentage": forward_percentage
#                 }
    
# # # face_image_inference("Isuru Alagiyawanna", 'data/facedb/Isuru Alagiyawanna/IMG-20240804-WA0009.jpg')
# # face_image_inference("Akshay Kumar", 'data/test_images/qqq.jpg')
# # video_face_inference("Isuru Alagiyawanna", is_vis=True)


# last oneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

# import cv2
# # import cv2 # cv2 import already exists as cv, no need to import again as cv2
# import time # cv2 is already imported as cv, using cv.
# import numpy as np
# import pandas as pd
# import yaml
# import pymongo
# import mediapipe as mp
# import faiss
# import glob
# import os # os was imported twice, removed one
# from deepface import DeepFace
# from datetime import datetime, timedelta
# # import os # os already imported

# # --- DEFINE BASE DIRECTORY AND DEFAULT PATHS ---
# # Get the directory where the current script (face_monitoring_inference.py) is located
# CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Go one level up from 'src' to 'USER_VARIFICATION_SERVICE' directory (or your project root)
# # Adjust '..' if your 'models' and 'data' folders are at a different relative level
# PROJECT_ROOT_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # This assumes 'src' is directly under project root

# DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
# DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')

# DEFAULT_FACE_INDEX_PATH = os.path.join(DEFAULT_MODELS_DIR, 'face_index')
# DEFAULT_FACE_DETAILS_PATH = os.path.join(DEFAULT_MODELS_DIR, 'face_details.npz')
# DEFAULT_FACE_IMAGE_DIR = os.path.join(DEFAULT_DATA_DIR, 'facedb', '*', '*.jpg') # For building the index
# DEFAULT_TEMP_DIR = os.path.join(DEFAULT_DATA_DIR, 'temp_dir') # For video_face_inference

# # Ensure temp_dir exists for video_face_inference
# if not os.path.exists(DEFAULT_TEMP_DIR):
#     os.makedirs(DEFAULT_TEMP_DIR)
#     print(f"--- [SETUP] Created temporary directory: {DEFAULT_TEMP_DIR}")
# # --- END OF DEFAULT PATH DEFINITIONS ---


# with open('secrets.yaml') as f: # This secrets.yaml should be accessible from where app.py is run
#     secrets = yaml.load(f, Loader=yaml.FullLoader)

# os.environ["MONGO_DB_URI"] = secrets['MONGO_DB_URI']

# try:
#     client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
#     db = client['Elearning'] # Make sure your DB name is correct
#     ffeatures_collection = db['ffeatures'] # Make sure collection name is correct
#     print("Connected to MongoDB")

# except Exception as e:
#     print(f"MongoDB Connection Error: {e}")

# face_mesh = mp.solutions.face_mesh.FaceMesh(
#     max_num_faces=1, # Optimized for single face if that's the usual case
#     refine_landmarks=True, # For more accurate landmarks like iris (if needed for future features)
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(
#     color=(128, 0, 128),
#     circle_radius=1,
#     thickness=1 # Adjusted thickness
# )
# # p_face_mesh = mp.solutions.face_mesh # This variable was not used, can be removed

# models = [
#     "VGG-Face",
#     "Facenet",
#     "Facenet512", # Used in extract_face_information_for_db
#     "OpenFace",   # Used in extract_face_information_for_inference (models[3] if list unchanged)
#     "DeepFace",
#     "DeepID",
#     "ArcFace",
#     "Dlib",
#     "SFace",
#     "GhostFaceNet",
# ]
# # Ensure the model selected by index (e.g., models[3]) matches the expected embedding dimension (e.g., 128 for OpenFace)
# SELECTED_INFERENCE_MODEL = models[3] # OpenFace (index 3) -> 128d
# SELECTED_DB_MODEL = models[2]        # Facenet512 (index 2) -> 512d
# # IMPORTANT: If DB model and inference model have different dimensions, this will cause issues
# # For consistency and to match d=128 for Faiss index, let's use OpenFace for both or ensure DB model is also 128d
# # For now, assuming Faiss index is built with d=128, so inference model should also be 128d.
# # Let's assume extract_face_information_for_db also uses a 128d model for consistency with the Faiss index d=128.
# # If extract_face_information_for_db MUST use Facenet512, then Faiss index 'd' must be 512.
# # We will proceed assuming d=128 for Faiss, so both functions should use a 128d model like OpenFace.

# DB_MODEL_FOR_EMBEDDINGS = SELECTED_INFERENCE_MODEL # e.g., OpenFace (128d)
# INFERENCE_MODEL_FOR_EMBEDDINGS = SELECTED_INFERENCE_MODEL # e.g., OpenFace (128d)
# FAISS_DIMENSION = 128 # Must match the output dimension of the model used for embeddings


# def head_pose_inference(
#         image,
#         image_flag=False
# ):
#     # start = time.time() # Moved FPS calculation outside if not needed for every call

#     if image_flag:
#         # Assuming image is already BGR if coming from cv2.imread
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     else:
#         # Flip image then convert to RGB
#         rgb_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

#     rgb_image.flags.writeable = False
#     results = face_mesh.process(rgb_image)
#     # image.flags.writeable = True # This was on the input 'image', should be on a copy or the processed one
#     # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) # This was on the input 'image'

#     # Create a writable copy for drawing
#     annotated_image = image.copy() # Work on a copy of the original BGR image
#     annotated_image.flags.writeable = True


#     img_h, img_w, _ = annotated_image.shape # Use annotated_image's shape
#     texts = []
#     face_centroids_coords = [] # Renamed for clarity

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks: # Should be only one if max_num_faces=1
#             face_2d = []
#             face_3d = []
#             # Keypoints for PnP as used in many examples
#             # Nose tip (1), Chin (199), Left eye left corner (33), Right eye right corner (263), Left Mouth corner (61), Right mouth corner (291)
#             keypoint_indices = [1, 199, 33, 263, 61, 291]
#             nose_2d, nose_3d = None, None # Initialize

#             for idx, lm in enumerate(face_landmarks.landmark):
#                 if idx in keypoint_indices:
#                     if idx == 1: # Nose tip
#                         nose_2d = (lm.x * img_w, lm.y * img_h)
#                         # Estimate Z for nose_3d. Adjust factor if needed.
#                         nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w * 0.7) # Adjusted Z scaling
#                     x, y = int(lm.x * img_w), int(lm.y * img_h)
#                     face_2d.append([x, y])
#                     # Using lm.z * img_w might be a common heuristic for Z.
#                     # The actual scale of lm.z is relative and needs calibration for precise 3D.
#                     face_3d.append([x, y, lm.z * img_w * 0.7]) # Adjusted Z scaling

#             if len(face_2d) != len(keypoint_indices) or nose_2d is None:
#                 # print("--- [HEAD_POSE_WARN] Not enough keypoints detected for PnP.")
#                 continue # Skip this face if not enough points

#             face_2d = np.array(face_2d, dtype=np.float64)
#             face_3d = np.array(face_3d, dtype=np.float64)

#             # Calculate face centroid from all landmarks for rough center
#             all_xs = [landmark.x * img_w for landmark in face_landmarks.landmark]
#             all_ys = [landmark.y * img_h for landmark in face_landmarks.landmark]
#             if all_xs and all_ys:
#                  face_centroids_coords.append((np.mean(all_xs), np.mean(all_ys)))


#             focal_length = img_w # Simplified focal length
#             cam_matrix = np.array([
#                 [focal_length, 0, img_w / 2],
#                 [0, focal_length, img_h / 2], # Corrected: img_h / 2 for y-center
#                 [0, 0, 1]
#             ])
#             distortion_matrix = np.zeros((4, 1), dtype=np.float64)

#             success, rotation_vec, translation_vec = cv2.solvePnP(
#                 face_3d, face_2d, cam_matrix, distortion_matrix
#             )

#             if not success:
#                 # print("--- [HEAD_POSE_WARN] solvePnP failed.")
#                 continue

#             rmat, _ = cv2.Rodrigues(rotation_vec)
#             # Decompose rotation matrix to Euler angles
#             # angles format: [pitch, yaw, roll]
#             sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
#             singular = sy < 1e-6
#             if not singular:
#                 x_angle = np.arctan2(rmat[2, 1], rmat[2, 2])
#                 y_angle = np.arctan2(-rmat[2, 0], sy)
#                 z_angle = np.arctan2(rmat[1, 0], rmat[0, 0])
#             else:
#                 x_angle = np.arctan2(-rmat[1, 2], rmat[1, 1])
#                 y_angle = np.arctan2(-rmat[2, 0], sy)
#                 z_angle = 0
            
#             # Convert radians to degrees
#             pitch_deg = np.degrees(x_angle)
#             yaw_deg = np.degrees(y_angle)
#             # roll_deg = np.degrees(z_angle) # Roll not typically used for "looking direction"

#             text = "Forward"
#             if yaw_deg < -10: # Looking left (negative yaw)
#                 text = "Looking Left"
#             elif yaw_deg > 10: # Looking right (positive yaw)
#                 text = "Looking Right"
#             elif pitch_deg < -10: # Looking down (negative pitch)
#                 text = "Looking Down"
#             elif pitch_deg > 10: # Looking up (positive pitch)
#                 text = "Looking Up"
#             texts.append(text)

#             # For drawing line (optional, can be removed if not needed in final image)
#             # nose_3d_projection, _ = cv2.projectPoints(
#             #     np.array([nose_3d]), rotation_vec, translation_vec, cam_matrix, distortion_matrix
#             # )
#             # p1 = (int(nose_2d[0]), int(nose_2d[1]))
#             # p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
#             # cv2.line(annotated_image, p1, p2, (0, 0, 255), 2)
#             # cv2.putText(annotated_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Draw face mesh (optional)
#             # mp_drawing.draw_landmarks(
#             #     image=annotated_image,
#             #     landmark_list=face_landmarks,
#             #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS, # Corrected constant
#             #     landmark_drawing_spec=None, # Do not draw individual landmarks
#             #     connection_drawing_spec=drawing_spec
#             # )
#     # else:
#         # print("--- [HEAD_POSE_INFO] No face landmarks detected in head_pose_inference.")


#     # end = time.time()
#     # totalTime = end - start
#     # if totalTime > 0:
#     #     fps = 1 / totalTime
#     #     # print("FPS (head_pose_inference): ", fps)
#     #     # cv2.putText(annotated_image, f'FPS: {int(fps)}', (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


#     return annotated_image, texts, face_centroids_coords


# def extract_face_information_for_db(img_path):
#     # This function is used to build the Faiss index.
#     # Ensure model and output dimension match FAISS_DIMENSION
#     # print(f"--- [DB_EXTRACT_DEBUG] Extracting for DB: {img_path} using model {DB_MODEL_FOR_EMBEDDINGS}")
#     try:
#         # Using enforce_detection=True for DB images is good to ensure quality.
#         face_objs = DeepFace.represent(
#             img_path=img_path,
#             model_name=DB_MODEL_FOR_EMBEDDINGS, # Should be 128d for consistency
#             enforce_detection=True, # Enforce for DB to get only good faces
#             detector_backend='retinaface' # Or another robust detector
#         )
#     except ValueError as ve: # Catch error if no face is detected with enforce_detection=True
#         print(f"--- [DB_EXTRACT_WARN] No face detected or error in DeepFace.represent for DB image {img_path}: {ve}")
#         return None, None, None, None
#     except Exception as e:
#         print(f"--- [DB_EXTRACT_ERROR] General error in DeepFace.represent for DB image {img_path}: {e}")
#         return None, None, None, None

#     img_path_norm = img_path.replace("\\", "/")
#     user_name = os.path.basename(os.path.dirname(img_path_norm)) # More robust way to get person_name

#     # DeepFace.represent with enforce_detection=True returns a list of dicts,
#     # each dict for one detected face. For DB, we usually want one face per image.
#     if not face_objs or not isinstance(face_objs, list) or len(face_objs) == 0:
#         print(f"--- [DB_EXTRACT_WARN] No face objects returned for DB image {img_path}")
#         return None, None, None, None

#     if len(face_objs) > 1:
#         print(f"--- [DB_EXTRACT_WARN] Multiple faces ({len(face_objs)}) detected in DB image {img_path}. Using the first one.")
#         # Optionally, add logic to select the largest face or most central one.

#     face_obj = face_objs[0] # Use the first detected face

#     # Check for expected keys based on the model's output
#     # For OpenFace, common keys are 'embedding', 'facial_area', 'face_confidence'
#     # For some versions/models, it might be 'representation' for embedding.
#     if 'embedding' not in face_obj or 'facial_area' not in face_obj:
#         print(f"--- [DB_EXTRACT_WARN] Missing 'embedding' or 'facial_area' in face_obj for {img_path}. Keys: {face_obj.keys()}")
#         return None, None, None, None

#     embedding = face_obj['embedding']
#     facial_area_dict = face_obj['facial_area']
#     # 'face_confidence' might not always be present or highly reliable from represent()
#     face_confidence = face_obj.get('face_confidence', 1.0) # Default to 1.0 if not present

#     if not all(k in facial_area_dict for k in ('x', 'y', 'w', 'h')):
#         print(f"--- [DB_EXTRACT_WARN] Facial area keys missing in face_obj for {img_path}.")
#         return None, None, None, None

#     x, y, w, h = facial_area_dict['x'], facial_area_dict['y'], facial_area_dict['w'], facial_area_dict['h']

#     # print(f"--- [DB_EXTRACT_DEBUG] Successfully extracted for DB: user={user_name}, emb_len={len(embedding)}")
#     return embedding, face_confidence, (x, y, w, h), user_name


# def build_face_embedding_index(
#         d=FAISS_DIMENSION, # Use the globally defined dimension
#         face_index_path=DEFAULT_FACE_INDEX_PATH,
#         face_image_dir=DEFAULT_FACE_IMAGE_DIR,
#         face_details_path=DEFAULT_FACE_DETAILS_PATH
# ):
#     current_working_dir = os.getcwd()
#     print(f"--- [BUILD_INDEX_DEBUG] Current Working Directory (where app.py/script is run): {current_working_dir}")
#     print(f"--- [BUILD_INDEX_DEBUG] Checking for face_index_path: {face_index_path}")
#     print(f"--- [BUILD_INDEX_DEBUG] Checking for face_details_path: {face_details_path}")

#     index_exists = os.path.exists(face_index_path)
#     details_exist = os.path.exists(face_details_path)
#     print(f"--- [BUILD_INDEX_DEBUG] Does face_index exist? {index_exists}")
#     print(f"--- [BUILD_INDEX_DEBUG] Does face_details exist? {details_exist}")

#     if not index_exists or not details_exist:
#         print(f"--- [BUILD_INDEX_INFO] Index files not found or incomplete. Building new index...")
#         print(f"--- [BUILD_INDEX_INFO] Building index from image directory: {face_image_dir}")

#         # Check if face_image_dir is valid and contains images
#         image_paths = glob.glob(face_image_dir)
#         if not image_paths:
#             print(f"--- [BUILD_INDEX_ERROR] No images found in directory: {face_image_dir}. Cannot build index.")
#             return None, None, None, None # Return None if no images to build from

#         faiss_index = faiss.index_factory(d, "Flat", faiss.METRIC_L2) # Using METRIC_L2 (Euclidean) is common
#         # faiss.METRIC_INNER_PRODUCT is also an option, ensure normalization if using it.

#         embeddings_list = [] # Renamed for clarity
#         user_names_list = []
#         facial_areas_list = []
#         face_confidences_list = []

#         total_images = len(image_paths)
#         print(f"--- [BUILD_INDEX_INFO] Found {total_images} images to process for the index.")

#         for idx, img_path in enumerate(image_paths):
#             emb, face_confidence, facial_area, user_name = extract_face_information_for_db(img_path)
#             if emb is not None and len(emb) == d: # Ensure embedding is not None and has correct dimension
#                 embeddings_list.append(emb)
#                 user_names_list.append(user_name)
#                 facial_areas_list.append(facial_area)
#                 face_confidences_list.append(face_confidence)
#             # else:
#                 # print(f"--- [BUILD_INDEX_WARN] Skipping image {img_path} for index (no valid embedding).")


#             if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
#                 print(f"--- [BUILD_INDEX_PROGRESS] Processed {idx + 1}/{total_images} images for index.")
        
#         if not embeddings_list:
#             print(f"--- [BUILD_INDEX_ERROR] No valid embeddings extracted from any images. Cannot create Faiss index.")
#             # Attempt to create empty files so it doesn't try to rebuild every time if this is intentional (no users yet)
#             # This part is tricky; if no users, maybe an empty index IS the desired state.
#             # For now, let's return None, which might cause rebuild on next app start if files are not created.
#             # To prevent rebuild, one could save an empty index and empty npz.
#             try:
#                 # Save an empty index
#                 empty_faiss_index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
#                 faiss.write_index(empty_faiss_index, face_index_path)
#                 # Save empty details
#                 np.savez(
#                     face_details_path,
#                     user_names=np.array([]),
#                     facial_areas=np.array([]).reshape(0,4), # ensure correct shape for empty array
#                     face_confidences=np.array([])
#                 )
#                 print(f"--- [BUILD_INDEX_INFO] Saved empty index and details files as no embeddings were found.")
#                 return empty_faiss_index, [], [], []
#             except Exception as e_save_empty:
#                 print(f"--- [BUILD_INDEX_ERROR] Could not save empty index/details: {e_save_empty}")
#                 return None, None, None, None


#         embeddings_np = np.asarray(embeddings_list).astype('float32')
#         if embeddings_np.ndim == 1: # Handle case of single embedding
#             embeddings_np = embeddings_np.reshape(1, -1)
            
#         # Normalize embeddings if using METRIC_INNER_PRODUCT or for cosine similarity with L2
#         # faiss.normalize_L2(embeddings_np) # Crucial for cosine similarity with IP or L2

#         faiss_index.add(embeddings_np)
#         faiss.write_index(faiss_index, face_index_path)
#         print(f"--- [BUILD_INDEX_INFO] Faiss index with {faiss_index.ntotal} vectors saved to {face_index_path}")

#         np.savez(
#             face_details_path,
#             user_names=np.array(user_names_list), # Save as numpy arrays
#             facial_areas=np.array(facial_areas_list),
#             face_confidences=np.array(face_confidences_list)
#         )
#         print(f"--- [BUILD_INDEX_INFO] Face details saved to {face_details_path}")
        
#         return faiss_index, user_names_list, facial_areas_list, face_confidences_list

#     else:
#         print(f"--- [BUILD_INDEX_INFO] Loading existing index from: {face_index_path}")
#         try:
#             faiss_index = faiss.read_index(face_index_path)
#             face_details_data = np.load(face_details_path, allow_pickle=True) # allow_pickle for safety
#             user_names_loaded = list(face_details_data['user_names']) # Convert back to list if saved as np.array
#             facial_areas_loaded = list(face_details_data['facial_areas'])
#             face_confidences_loaded = list(face_details_data['face_confidences'])
#             print(f"--- [BUILD_INDEX_INFO] Successfully loaded index with {faiss_index.ntotal} vectors and details for {len(user_names_loaded)} entries.")
#             return faiss_index, user_names_loaded, facial_areas_loaded, face_confidences_loaded
#         except Exception as e_load:
#             print(f"--- [BUILD_INDEX_ERROR] Error loading existing index/details files: {e_load}. Attempting to rebuild.")
#             # Invalidate paths to force rebuild
#             if os.path.exists(face_index_path): os.remove(face_index_path)
#             if os.path.exists(face_details_path): os.remove(face_details_path)
#             return build_face_embedding_index(d, face_index_path, face_image_dir, face_details_path) # Recursive call to rebuild

# def extract_face_information_for_inference(img_path):
#     print(f"--- [EXTRACT_INFERENCE_DEBUG] Starting extraction for: {img_path} using model {INFERENCE_MODEL_FOR_EMBEDDINGS}")
#     try:
#         face_objs = DeepFace.represent(
#             img_path=img_path,
#             model_name=INFERENCE_MODEL_FOR_EMBEDDINGS, # Should be 128d
#             enforce_detection=False, # Don't enforce, handle no-face case
#             detector_backend='retinaface'
#         )
#         print(f"--- [EXTRACT_INFERENCE_DEBUG] RAW face_objs from DeepFace.represent: {face_objs}")
#     except Exception as e_repr:
#         print(f"--- [EXTRACT_INFERENCE_ERROR] Exception in DeepFace.represent for inference: {e_repr}")
#         import traceback
#         traceback.print_exc()
#         return [], [], []

#     embeddings = []
#     facial_areas = []
#     face_confidences_from_detection = [] # Renamed for clarity

#     if not face_objs or not isinstance(face_objs, list) or len(face_objs) == 0:
#         print(f"--- [EXTRACT_INFERENCE_WARN] No face objects returned by DeepFace.represent for {img_path}")
#     else:
#         print(f"--- [EXTRACT_INFERENCE_DEBUG] DeepFace.represent (inference) returned {len(face_objs)} face object(s).")
#         for i, face_obj in enumerate(face_objs):
#             if not isinstance(face_obj, dict):
#                 print(f"--- [EXTRACT_INFERENCE_WARN] Face object {i} is not a dictionary: {type(face_obj)}")
#                 continue

#             # Default confidence if not provided by represent() for a specific model
#             # For OpenFace, face_confidence from detection might be low or not the primary metric.
#             # Embedding similarity is more important.
#             face_confidence_val = face_obj.get('face_confidence', 0.0) # Default to 0.0 if no confidence key
#             # print(f"--- [EXTRACT_INFERENCE_DEBUG] Processing face object {i}: Detection Confidence: {face_confidence_val}")

#             # Check for essential keys. For OpenFace, it's usually 'embedding' and 'facial_area'.
#             if 'embedding' in face_obj and 'facial_area' in face_obj:
#                 emb = face_obj['embedding']
#                 facial_area_dict = face_obj['facial_area']

#                 if len(emb) != FAISS_DIMENSION:
#                     print(f"--- [EXTRACT_INFERENCE_WARN] Embedding for face {i} has dimension {len(emb)}, expected {FAISS_DIMENSION}. Skipping.")
#                     continue

#                 if all(k in facial_area_dict for k in ('x', 'y', 'w', 'h')):
#                     x, y, w, h = facial_area_dict['x'], facial_area_dict['y'], facial_area_dict['w'], facial_area_dict['h']
#                     embeddings.append(emb)
#                     facial_areas.append((x, y, w, h))
#                     face_confidences_from_detection.append(face_confidence_val) # This is detection confidence
#                     # print(f"--- [EXTRACT_INFERENCE_DEBUG] Appended embedding for face {i}. Detection confidence: {face_confidence_val}")
#                 else:
#                     print(f"--- [EXTRACT_INFERENCE_WARN] Facial area keys missing for face object {i}. Keys: {facial_area_dict.keys()}")
#             else:
#                 print(f"--- [EXTRACT_INFERENCE_WARN] Expected keys ('embedding', 'facial_area') missing in face object {i}. Keys: {face_obj.keys()}")
    
#     print(f"--- [EXTRACT_INFERENCE_DEBUG] Returning {len(embeddings)} embeddings, {len(face_confidences_from_detection)} detection_confidences, {len(facial_areas)} facial_areas.")
#     return embeddings, face_confidences_from_detection, facial_areas


# def search_face_in_db(
#         img_path,
#         face_index_path=DEFAULT_FACE_INDEX_PATH,
#         face_details_path=DEFAULT_FACE_DETAILS_PATH
# ):
#     print(f"--- [SEARCH_DB_DEBUG] Starting search_face_in_db for: {img_path}")
#     faiss_idx, user_names_in_db, _, _ = build_face_embedding_index( # Using faiss_idx to avoid name clash
#         d=FAISS_DIMENSION, # Explicitly pass dimension
#         face_index_path=face_index_path,
#         face_details_path=face_details_path
#     )

#     if faiss_idx is None:
#         print(f"--- [SEARCH_DB_ERROR] Faiss index is None after build_face_embedding_index. Cannot proceed.")
#         return [], [], []
#     if user_names_in_db is None or len(user_names_in_db) == 0:
#         # This case should ideally be handled by build_face_embedding_index returning an empty (but valid) index
#         print(f"--- [SEARCH_DB_INFO] No user names loaded from index (index might be empty or failed to load).")
#         # return [], [], [] # If index is truly empty, ntotal will be 0, search will handle it.

#     print(f"--- [SEARCH_DB_DEBUG] Index available. Total entries in Faiss index: {faiss_idx.ntotal}.")

#     embeddings, detection_confidences, facial_areas = extract_face_information_for_inference(img_path)
#     print(f"--- [SEARCH_DB_DEBUG] Extracted from inference: {len(embeddings)} embeddings, detection_confidences: {detection_confidences}")

#     retrieved_user_names_final = []
#     retrieved_facial_areas_final = []
#     retrieved_match_confidences_final = []

#     if not embeddings:
#         print(f"--- [SEARCH_DB_WARN] No embeddings extracted from '{img_path}'. Cannot search in DB.")
#         return [], [], []

#     if faiss_idx.ntotal == 0:
#         print(f"--- [SEARCH_DB_WARN] Faiss index is empty (ntotal=0). No users to match against.")
#         return [], [], []

#     for idx, emb_to_search in enumerate(embeddings):
#         # print(f"--- [SEARCH_DB_DEBUG] Processing embedding {idx + 1}/{len(embeddings)}. Detection confidence: {detection_confidences[idx]}")

#         # Threshold for considering a detected face for searching (detection confidence)
#         # This is different from Faiss match confidence.
#         DETECTION_CONF_THRESHOLD = 0.3 # Lower this if faces are missed by DeepFace's detector
#         if detection_confidences[idx] < DETECTION_CONF_THRESHOLD:
#             # print(f"--- [SEARCH_DB_WARN] Detection confidence {detection_confidences[idx]} < {DETECTION_CONF_THRESHOLD}. Skipping Faiss search for this face.")
#             continue
        
#         # print(f"--- [SEARCH_DB_DEBUG] Detection confidence {detection_confidences[idx]} >= {DETECTION_CONF_THRESHOLD}. Proceeding with Faiss search.")
#         emb_np = np.array(emb_to_search).reshape(1, -1).astype('float32')
#         # faiss.normalize_L2(emb_np) # Normalize if index uses cosine similarity (IP or L2 normalized)

#         k_neighbors = min(5, faiss_idx.ntotal) # Search for at most 5, or fewer if index is small
#         if k_neighbors == 0: # Should not happen if ntotal check passed, but good for safety
#             # print(f"--- [SEARCH_DB_WARN] k_neighbors is 0, cannot search.")
#             continue
            
#         # print(f"--- [SEARCH_DB_DEBUG] Searching for {k_neighbors} neighbors in Faiss index (ntotal={faiss_idx.ntotal}).")
        
#         distances, indices = faiss_idx.search(emb_np, k_neighbors)
#         # print(f"--- [SEARCH_DB_DEBUG] Faiss search results: Distances (D): {distances}, Indices (I): {indices}")

#         # distances are L2 squared if METRIC_L2. Lower is better.
#         # For METRIC_INNER_PRODUCT (cosine sim after normalization), higher is better.
#         # We are using METRIC_L2. Convert to similarity: e.g., 1 / (1 + distance) or exp(-distance)
#         # Or, use a distance threshold.

#         valid_indices_mask = (indices != -1) & (indices < len(user_names_in_db)) # Also check bounds
        
#         # Flatten and apply mask
#         flat_indices = indices.flatten()[valid_indices_mask.flatten()]
#         flat_distances = distances.flatten()[valid_indices_mask.flatten()]

#         if len(flat_indices) == 0:
#             # print(f"--- [SEARCH_DB_WARN] No valid neighbors found in Faiss search for embedding {idx}.")
#             continue

#         candidate_user_names = [user_names_in_db[i_val] for i_val in flat_indices]
#         # print(f"--- [SEARCH_DB_DEBUG] Candidate user names from Faiss: {candidate_user_names}")

#         if candidate_user_names:
#             # Find the most frequent user among top k
#             from collections import Counter
#             count = Counter(candidate_user_names)
#             majority_user_name = count.most_common(1)[0][0]
            
#             # Calculate average distance for the majority user
#             # (Lower distance means better match for L2)
#             distances_for_majority_user = [
#                 dist for i_val, dist in zip(flat_indices, flat_distances)
#                 if user_names_in_db[i_val] == majority_user_name
#             ]
            
#             if distances_for_majority_user:
#                 avg_distance_for_majority = np.mean(distances_for_majority_user)
#                 # print(f"--- [SEARCH_DB_DEBUG] Majority user: {majority_user_name}, Avg. Faiss Match Distance (L2): {avg_distance_for_majority}")

#                 # Define a threshold for L2 distance. This needs tuning.
#                 # OpenFace L2 distances are often < 1.0 for matches. 0.6-0.8 might be a starting point.
#                 L2_DISTANCE_THRESHOLD = 0.8 # Tune this value!
#                 if avg_distance_for_majority <= L2_DISTANCE_THRESHOLD:
#                     # Convert distance to a pseudo-confidence (0-1, higher is better)
#                     # Simple inversion: 1.0 means perfect match (dist=0), threshold dist maps to ~0.5
#                     match_confidence_pseudo = max(0, 1.0 - (avg_distance_for_majority / (L2_DISTANCE_THRESHOLD * 2)))

#                     retrieved_match_confidences_final.append(np.round(match_confidence_pseudo, 3))
#                     retrieved_facial_areas_final.append(facial_areas[idx])
#                     retrieved_user_names_final.append(majority_user_name)
#                     # print(f"--- [SEARCH_DB_INFO] MATCH FOUND: User={majority_user_name}, PseudoConfidence={match_confidence_pseudo}, L2Dist={avg_distance_for_majority}")
#                 # else:
#                     # print(f"--- [SEARCH_DB_INFO] No strong match: User={majority_user_name}, L2Dist={avg_distance_for_majority} > threshold {L2_DISTANCE_THRESHOLD}")
#             # else:
#                 # print(f"--- [SEARCH_DB_WARN] No distances found for majority user {majority_user_name}.")
#         # else:
#             # print(f"--- [SEARCH_DB_WARN] candidate_user_names list is empty after Faiss search for embedding {idx}.")
            
#     print(f"--- [SEARCH_DB_DEBUG] search_face_in_db returning: users='{retrieved_user_names_final}', match_confidences='{retrieved_match_confidences_final}'")
#     return retrieved_user_names_final, retrieved_facial_areas_final, retrieved_match_confidences_final


# def eculedian_distance(x1, y1, x2, y2):
#     return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# def face_image_inference(
#         username, # Expected username
#         face_image_path
# ):
#     print(f"--- [FACE_INFERENCE_DEBUG] Starting face_image_inference for user: {username}, image: {face_image_path}")

#     img = cv2.imread(face_image_path)
#     if img is None:
#         print(f"--- [FACE_INFERENCE_ERROR] Could not read image from {face_image_path}")
#         raise ValueError(f"Could not read image: {face_image_path}") # Let app.py catch this

#     img_for_pose = img.copy() # Use a copy for head pose
#     final_head_pose_text = "Unknown" # Default
    
#     print(f"--- [FACE_INFERENCE_DEBUG] Calling head_pose_inference...")
#     try:
#         # annotated_image_from_pose, texts_from_pose, face_centroids_from_pose = head_pose_inference(img_for_pose, image_flag=True)
#         _, texts_from_pose, face_centroids_from_pose = head_pose_inference(img_for_pose, image_flag=True) # We only need texts and centroids
#         # print(f"--- [FACE_INFERENCE_DEBUG] head_pose_inference returned: texts='{texts_from_pose}', face_centroids_count={len(face_centroids_from_pose) if face_centroids_from_pose else 0}")
#         if texts_from_pose: # If head pose detected, use the first one detected
#             final_head_pose_text = texts_from_pose[0] 
#     except Exception as hp_e:
#         print(f"--- [FACE_INFERENCE_ERROR] Exception in head_pose_inference: {hp_e}")
#         # final_head_pose_text remains "Unknown"

#     print(f"--- [FACE_INFERENCE_DEBUG] Calling search_face_in_db...")
#     # try:
#     retrieved_user_names, retrieved_facial_areas, retrieved_match_confidences = search_face_in_db(face_image_path)
#     # print(f"--- [FACE_INFERENCE_DEBUG] search_face_in_db returned: users='{retrieved_user_names}', confidences='{retrieved_match_confidences}'")
#     # except Exception as sdb_e: # search_face_in_db should handle its own errors and return empty lists
#     #     print(f"--- [FACE_INFERENCE_ERROR] Exception in search_face_in_db call: {sdb_e}")
#     #     retrieved_user_names = []
    
#     final_det_username_api_response = "N/A" # Default for API response

#     if not retrieved_user_names:
#         print(f"--- [FACE_INFERENCE_WARN] No users reliably retrieved from DB for {face_image_path}. Inserting 'N/A' record.")
#         timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         ffeatures_collection.insert_one({
#             "exp_username": username,
#             "det_username": "N/A_NoMatchInDB", # More specific N/A
#             "head_pose": final_head_pose_text, # Use head pose if available
#             "face_confidence": 0.0, # No match confidence
#             "timestamp": timestamp_now
#         })
#     else:
#         # Assuming we process the first (and hopefully only) strong match from search_face_in_db
#         # If multiple faces were in input image and multiple matches, this logic might need refinement.
#         # For now, let's assume search_face_in_db returns a single best match if any.
        
#         # We need to associate the head_pose with the correct detected face if multiple faces.
#         # Current head_pose_inference gives one pose list for the whole image.
#         # If search_face_in_db can return multiple users for multiple faces in image,
#         # we need to match head_pose to the correct face.
#         # For simplicity, if only one user is retrieved, we use the global head_pose.
        
#         # Let's iterate through all retrieved users, though ideally search_face_in_db
#         # should be refined to return the best single candidate if multiple faces in input.
#         # Or, this loop should handle associating head pose to the specific facial_area.
        
#         # For now, this loop assumes retrieved_user_names corresponds to faces in order of detection.
#         # And head_pose_inference provides poses in a corresponding order.
#         # This is a simplification and might need robust handling for multi-face scenarios.
        
#         best_match_idx = -1
#         highest_confidence = -1.0

#         for i in range(len(retrieved_user_names)):
#             if retrieved_user_names[i] == username and retrieved_match_confidences[i] > highest_confidence:
#                 highest_confidence = retrieved_match_confidences[i]
#                 best_match_idx = i
        
#         if best_match_idx != -1:
#             # A match for the expected user was found
#             verified_user_name = retrieved_user_names[best_match_idx]
#             verified_match_conf = retrieved_match_confidences[best_match_idx]
#             # x, y, w, h = retrieved_facial_areas[best_match_idx] # For drawing, if needed

#             print(f"--- [FACE_INFERENCE_INFO] User {username} VERIFIED as {verified_user_name} with match_confidence {verified_match_conf}")
#             final_det_username_api_response = verified_user_name 
#             # Head pose for this specific verified face (simplistic: assumes one face or first head pose)
#             # More robust: use facial_area of verified face to find closest centroid from head_pose_inference
            
#             head_pose_for_verified_user = final_head_pose_text # Use the overall image head pose for now.
#             if face_centroids_from_pose and texts_from_pose and best_match_idx < len(retrieved_facial_areas):
#                  verified_face_area_coords = retrieved_facial_areas[best_match_idx]
#                  verified_face_center_x = verified_face_area_coords[0] + verified_face_area_coords[2] // 2
#                  verified_face_center_y = verified_face_area_coords[1] + verified_face_area_coords[3] // 2
                 
#                  if face_centroids_from_pose: # Ensure it's not empty
#                     distances_to_centroids = [
#                         eculedian_distance(verified_face_center_x, verified_face_center_y, cent_x, cent_y)
#                         for cent_x, cent_y in face_centroids_from_pose
#                     ]
#                     if distances_to_centroids:
#                         closest_centroid_idx = np.argmin(distances_to_centroids)
#                         if closest_centroid_idx < len(texts_from_pose):
#                             head_pose_for_verified_user = texts_from_pose[closest_centroid_idx]
#                             final_head_pose_text = head_pose_for_verified_user # Update final head pose for API

#             timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             ffeatures_collection.insert_one({
#                 "exp_username": username,
#                 "det_username": verified_user_name,
#                 "head_pose": head_pose_for_verified_user,
#                 "face_confidence": float(verified_match_conf), # This is match confidence
#                 "timestamp": timestamp_now
#             })
#         else:
#             # No match for the *expected* user, or no strong matches at all.
#             # Log what was actually detected, if anything.
#             print(f"--- [FACE_INFERENCE_WARN] User {username} NOT verified. Best actual detections (if any): {retrieved_user_names}")
#             timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             if retrieved_user_names: # Log the strongest actual detection if one exists
#                 strongest_actual_detection_idx = np.argmax(retrieved_match_confidences)
#                 actual_det_user = retrieved_user_names[strongest_actual_detection_idx]
#                 actual_det_conf = retrieved_match_confidences[strongest_actual_detection_idx]
#                 # Head pose for this different user (simplistic)
#                 head_pose_for_actual_detection = final_head_pose_text # Use overall image head pose
#                 # (Add logic to associate with specific face if needed)

#                 ffeatures_collection.insert_one({
#                     "exp_username": username,
#                     "det_username": actual_det_user, # The user actually detected
#                     "head_pose": head_pose_for_actual_detection,
#                     "face_confidence": float(actual_det_conf),
#                     "timestamp": timestamp_now
#                 })
#                 # final_det_username_api_response remains "N/A" because expected user was not verified
#             else: # Nothing retrieved from DB at all (already handled by the first 'if not retrieved_user_names')
#                 # This block might be redundant if the first check handles it.
#                 # Kept for clarity that if expected user not found, default N/A.
#                  pass


#     print(f"--- [FACE_INFERENCE_DEBUG] face_image_inference returning: final_head_pose_text='{final_head_pose_text}', final_det_username_api_response='{final_det_username_api_response}'")
#     return final_head_pose_text, final_det_username_api_response


# def video_face_inference(
#         username,
#         is_vis=False
# ):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("--- [VIDEO_INFERENCE_ERROR] Cannot open camera.")
#         return

#     temp_image_path = os.path.join(DEFAULT_TEMP_DIR, "temp_video_frame.jpg")

#     while cap.isOpened():
#         success, img_from_cam = cap.read()
#         if not success:
#             print("--- [VIDEO_INFERENCE_WARN] Failed to grab frame from camera.")
#             break

#         # Process a copy for head pose, keep original for other processing
#         img_for_pose = img_from_cam.copy()
#         # Flip the image that will be saved and used for DeepFace, as webcam often mirrors
#         img_for_deepface = cv2.flip(img_from_cam, 1)
#         cv2.imwrite(temp_image_path, img_for_deepface)

#         # Head pose on the non-flipped image (as seen by user)
#         # annotated_image_display, texts_from_pose, face_centroids_from_pose = head_pose_inference(img_for_pose, image_flag=False) # image_flag=False for webcam
#         final_head_pose_text_video = "Unknown" # Default
#         try:
#             annotated_image_display, texts_from_pose, face_centroids_from_pose = head_pose_inference(img_for_pose, image_flag=False)
#             if texts_from_pose:
#                 final_head_pose_text_video = texts_from_pose[0]
#         except Exception as hp_e_vid:
#             print(f"--- [VIDEO_INFERENCE_ERROR] Exception in head_pose_inference (video): {hp_e_vid}")
#             annotated_image_display = img_for_pose # Show original if pose fails

#         # Face recognition on the (potentially flipped) saved image
#         retrieved_user_names, retrieved_facial_areas, retrieved_match_confidences = search_face_in_db(temp_image_path)
        
#         final_verified_user_video = "N/A" # Default

#         # Logic to find if the *expected* username is among the verified faces
#         best_match_idx_video = -1
#         highest_confidence_video = -1.0

#         for i in range(len(retrieved_user_names)):
#             if retrieved_user_names[i] == username and retrieved_match_confidences[i] > highest_confidence_video:
#                 highest_confidence_video = retrieved_match_confidences[i]
#                 best_match_idx_video = i
        
#         timestamp_now_video = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         if best_match_idx_video != -1:
#             # Expected user verified
#             verified_user_name_video = retrieved_user_names[best_match_idx_video]
#             verified_match_conf_video = retrieved_match_confidences[best_match_idx_video]
#             x, y, w, h = retrieved_facial_areas[best_match_idx_video] # Coords are for the image given to DeepFace (flipped)
            
#             final_verified_user_video = verified_user_name_video
#             # Associate head pose
#             head_pose_for_verified_user_video = final_head_pose_text_video
#             # (Add more robust association logic if multiple faces/poses detected)

#             ffeatures_collection.insert_one({
#                 "exp_username": username,
#                 "det_username": verified_user_name_video,
#                 "head_pose": head_pose_for_verified_user_video,
#                 "face_confidence": float(verified_match_conf_video),
#                 "timestamp": timestamp_now_video
#             })
#             # Draw on the display image (non-flipped). Coords need to be adjusted if display is different from processing.
#             # For simplicity, assuming DeepFace facial_area is on the flipped image.
#             # To draw on non-flipped, either re-run detection or transform coords.
#             # For now, let's draw on the image used for display (annotated_image_display)
#             # but facial_area is from the flipped image, so convert 'x' for drawing.
#             # img_width_disp = annotated_image_display.shape[1]
#             # x_display = img_width_disp - (x + w) # Convert x for flipped image

#             # cv2.rectangle(annotated_image_display, (x_display, y), (x_display + w, y + h), (0, 255, 0), 2)
#             # cv2.putText(annotated_image_display, f'User: {verified_user_name_video}', (x_display - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         elif retrieved_user_names: # Expected user not found, but *someone* was detected
#             strongest_actual_detection_idx_video = np.argmax(retrieved_match_confidences)
#             actual_det_user_video = retrieved_user_names[strongest_actual_detection_idx_video]
#             actual_det_conf_video = retrieved_match_confidences[strongest_actual_detection_idx_video]
#             # x_actual, y_actual, w_actual, h_actual = retrieved_facial_areas[strongest_actual_detection_idx_video]

#             ffeatures_collection.insert_one({
#                 "exp_username": username,
#                 "det_username": actual_det_user_video, # Log who was actually seen
#                 "head_pose": final_head_pose_text_video,
#                 "face_confidence": float(actual_det_conf_video),
#                 "timestamp": timestamp_now_video
#             })
#             # x_display_actual = annotated_image_display.shape[1] - (x_actual + w_actual)
#             # cv2.rectangle(annotated_image_display, (x_display_actual, y_actual), (x_display_actual + w_actual, y_actual + h_actual), (0, 0, 255), 2)
#         else: # No one reliably detected by search_face_in_db
#             ffeatures_collection.insert_one({
#                 "exp_username": username,
#                 "det_username": "N/A_NoMatchVideo",
#                 "head_pose": final_head_pose_text_video,
#                 "face_confidence": 0.0,
#                 "timestamp": timestamp_now_video
#             })

#         if is_vis:
#             # Add text overlay for verified user on the display image
#             display_text_user = f"Expected: {username}"
#             display_text_detected = f"Detected: {final_verified_user_video} (Match: {highest_confidence_video:.2f})" if best_match_idx_video != -1 else f"Detected: {retrieved_user_names[np.argmax(retrieved_match_confidences)] if retrieved_user_names else 'N/A'}"
#             display_text_pose = f"Pose: {final_head_pose_text_video}"

#             cv2.putText(annotated_image_display, display_text_user, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
#             cv2.putText(annotated_image_display, display_text_detected, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
#             cv2.putText(annotated_image_display, display_text_pose, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

#             cv2.imshow('Face Monitoring Inference - Live', annotated_image_display)
#             if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
#                 break
#     # else: # Loop finished naturally (e.g. camera closed)
#         # print("--- [VIDEO_INFERENCE_INFO] Video capture loop finished.")


#     cap.release()
#     cv2.destroyAllWindows()
#     print("--- [VIDEO_INFERENCE_INFO] Camera released and windows closed.")


# def face_analysis(
#         username,
#         x_min=1440 # Minutes
# ):
#     current_time = datetime.now()
#     current_time_minus_x = current_time - timedelta(minutes=x_min)

#     # Format for MongoDB query if timestamps are stored as strings
#     # If stored as datetime objects, no need to format query times this way
#     # query_time_gte = current_time_minus_x.strftime("%Y-%m-%d %H:%M:%S")
#     # query_time_lt = current_time.strftime("%Y-%m-%d %H:%M:%S")

#     # Assuming timestamps in DB are ISODate objects or comparable strings
#     data_user_cursor = ffeatures_collection.find({
#         "exp_username": username,
#         "timestamp": {
#             "$gte": current_time_minus_x.strftime("%Y-%m-%d %H:%M:%S"), # Query with string format
#             "$lt": current_time.strftime("%Y-%m-%d %H:%M:%S")
#         }
#     })
#     data_user_list = list(data_user_cursor)

#     if not data_user_list:
#         print(f"--- [FACE_ANALYSIS_INFO] No data found for user {username} in the last {x_min} minutes.")
#         return { # Return a default structure even if empty
#             "detected_percentage": "0.00 %",
#             "forward_percentage": "0.00 %",
#             "message": "No activity data found for the specified period."
#         }

#     data_user_df = pd.DataFrame(data_user_list)

#     if data_user_df.empty: # Should be caught by previous check, but good for safety
#          return {
#             "detected_percentage": "0.00 %",
#             "forward_percentage": "0.00 %",
#             "message": "No activity data (empty DataFrame)."
#         }

#     # Ensure columns exist before trying to use them
#     n_total = len(data_user_df)
    
#     n_forward = 0
#     if "head_pose" in data_user_df.columns:
#         n_forward = len(data_user_df[data_user_df["head_pose"] == "Forward"])
    
#     n_detected_correctly = 0
#     if "det_username" in data_user_df.columns:
#         n_detected_correctly = len(data_user_df[data_user_df["det_username"] == username])
    
#     detected_percentage = (n_detected_correctly / n_total) * 100 if n_total > 0 else 0
#     forward_percentage = (n_forward / n_total) * 100 if n_total > 0 else 0

#     return {
#         "detected_percentage": f"{detected_percentage:.2f} %",
#         "forward_percentage": f"{forward_percentage:.2f} %"
#     }

# # --- Example Usage (for testing directly) ---
# if __name__ == '__main__':
#     print("--- Running face_monitoring_inference.py directly for testing ---")
    
#     # Ensure your test user and image exist in the paths defined by DEFAULT_...
#     test_username = "purna" # Change to a user in your facedb
    
#     # Find a test image for this user
#     test_user_image_dir = os.path.join(DEFAULT_DATA_DIR, 'facedb', test_username)
#     test_image_files = glob.glob(os.path.join(test_user_image_dir, '*.jpg')) # or .png etc.
    
#     if not test_image_files:
#         print(f"--- [TEST_ERROR] No JPG images found for user '{test_username}' in {test_user_image_dir}")
#         print(f"--- [TEST_INFO] Please add images or check DEFAULT_FACE_IMAGE_DIR and test_username.")
#     else:
#         test_image_path_for_script = test_image_files[0] # Use the first image found
#         print(f"--- [TEST_INFO] Using test image: {test_image_path_for_script} for user: {test_username}")

#         # 1. Test building the index (it will use default paths)
#         # Delete existing index files to force a rebuild for testing
#         if os.path.exists(DEFAULT_FACE_INDEX_PATH): os.remove(DEFAULT_FACE_INDEX_PATH)
#         if os.path.exists(DEFAULT_FACE_DETAILS_PATH): os.remove(DEFAULT_FACE_DETAILS_PATH)
#         print(f"--- [TEST_INFO] Deleted existing index files (if any) to force rebuild.")
        
#         build_face_embedding_index() # This will print its own debug messages

#         # 2. Test face_image_inference
#         print(f"\n--- [TEST_INFO] Testing face_image_inference with user '{test_username}' and image '{test_image_path_for_script}' ---")
#         head_pose, det_user = face_image_inference(test_username, test_image_path_for_script)
#         print(f"--- [TEST_RESULT] face_image_inference: Head Pose='{head_pose}', Detected User='{det_user}'")

#         # 3. Test face_analysis (will use data inserted by face_image_inference if successful)
#         print(f"\n--- [TEST_INFO] Testing face_analysis for user '{test_username}' (last 60 minutes) ---")
#         analysis_result = face_analysis(test_username, x_min=60)
#         print(f"--- [TEST_RESULT] face_analysis: {analysis_result}")

#     # 4. Test video inference (optional, will open webcam)
#     # print("\n--- [TEST_INFO] Testing video_face_inference. Press ESC in the window to quit. ---")
#     # video_face_inference("purna", is_vis=True) # Change username if needed




# import cv2 as cv
# import cv2, time
# import numpy as np
# import pandas as pd
# import yaml, pymongo
# import mediapipe as mp
# import faiss, glob, os
# from deepface import DeepFace
# from datetime import datetime, timedelta

# with open('secrets.yaml') as f:
#     secrets = yaml.load(f, Loader=yaml.FullLoader)

# os.environ["MONGO_DB_URI"] = secrets['MONGO_DB_URI']

# try:
#     client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
#     db = client['Elearning']
#     ffeatures_collection = db['ffeatures']
#     print("Connected to MongoDB")
    
# except Exception as e:
#     print(e)

# face_mesh = mp.solutions.face_mesh.FaceMesh(
#                                             min_detection_confidence=0.5, 
#                                             min_tracking_confidence=0.5
#                                             )

# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(
#                                     color=(128,0,128),
#                                     circle_radius=1,
#                                     thickness=2
#                                     )
# p_face_mesh = mp.solutions.face_mesh

# models = [
#         "VGG-Face", 
#         "Facenet", 
#         "Facenet512", 
#         "OpenFace", 
#         "DeepFace", 
#         "DeepID", 
#         "ArcFace", 
#         "Dlib", 
#         "SFace",
#         "GhostFaceNet",
#         ]

# def head_pose_inference(
#                         image,
#                         image_flag = False
#                         ):
#     start = time.time()

#     if image_flag:
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     else:
#         image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) 
#     image.flags.writeable = False

#     results = face_mesh.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

#     img_h , img_w, img_c = image.shape
#     face_2d = []
#     face_3d = []

#     texts = []
#     face_centroids = []
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for idx, lm in enumerate(face_landmarks.landmark):
#                 if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
#                     if idx ==1:
#                         nose_2d = (lm.x * img_w,lm.y * img_h)
#                         nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
#                     x,y = int(lm.x * img_w),int(lm.y * img_h)

#                     face_2d.append([x,y])
#                     face_3d.append(([x,y,lm.z]))

#             face_2d = np.array(face_2d,dtype=np.float64)
#             face_3d = np.array(face_3d,dtype=np.float64)

#             face_centroid = np.mean(face_2d,axis=0)

#             focal_length = 1 * img_w
#             cam_matrix = np.array([[focal_length,0,img_h/2],
#                                   [0,focal_length,img_w/2],
#                                   [0,0,1]])
#             distortion_matrix = np.zeros((4,1),dtype=np.float64)
#             success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)

#             rmat,jac = cv2.Rodrigues(rotation_vec)
#             angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

#             x = angles[0] * 360
#             y = angles[1] * 360
#             z = angles[2] * 360

#             if y < -10:
#                 text="Looking Left"
#             elif y > 10:
#                 text="Looking Right"
#             elif x < -10:
#                 text="Looking Down"
#             elif x > 10:
#                 text="Looking Up"
#             else:
#                 text="Forward"
#             texts.append(text)
#             face_centroids.append(face_centroid)
#             nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

#             p1 = (int(nose_2d[0]),int(nose_2d[1]))
#             p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

#             cv2.line(image,p1,p2,(255,0,0),3)

#             cv2.putText(image,text,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),3)
#             cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#             cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#             cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#         end = time.time()
#         totalTime = end-start

#         fps = 1/totalTime
#         print("FPS: ",fps)

#         cv2.putText(image,f'FPS: {int(fps)}',(20,450),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

#         mp_drawing.draw_landmarks(
#                                 image=image,
#                                 landmark_list=face_landmarks,
#                                 connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#                                 connection_drawing_spec=drawing_spec,
#                                 landmark_drawing_spec=drawing_spec
#                                 )
#     return image, texts, face_centroids

# def extract_face_information_for_db(img_path):
#     face_objs = DeepFace.represent(
#                                 img_path = img_path,
#                                 model_name = models[2],
#                                 enforce_detection = False
#                                 )
#     img_path = img_path.replace("\\", "/")
#     user_name = img_path.split("/")[-2]

#     if len(face_objs) != 1:
#         if len(face_objs) == 0:
#             Warning(f"No faces detected in the image : {img_path}")
#         else:
#             Warning(f"Multiple faces detected in the image : {img_path}")
#         return None, None, None, None

#     else:
#         facial_area = face_objs[0]['facial_area']
#         embeddings = face_objs[0]['embedding']
#         face_confidence = face_objs[0]['face_confidence']
#         x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

#     return embeddings, face_confidence, (x, y, w, h), user_name

# def build_face_embedding_index(
#                                 d = 512,
#                                 face_index_path = 'models/face_index',
#                                 face_image_dir = 'data/facedb/*/*.jpg',
#                                 face_details_path = 'models/face_details.npz',
#                                 ):
#     if (not os.path.exists(face_index_path)) or (not os.path.exists(face_details_path)):
#         faiss_index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

#         embeddings = []
#         user_names = []
#         facial_areas = []
#         face_confidences = []
        
#         for idx, img_path in enumerate(glob.glob(face_image_dir)):
#             emb, face_confidence, facial_area, user_name = extract_face_information_for_db(img_path)
#             if emb is not None:
#                 embeddings.append(emb)
#                 user_names.append(user_name)
#                 facial_areas.append(facial_area)
#                 face_confidences.append(face_confidence)

#             if idx % 10 == 0:
#                 print(f"Processed {idx}/{len(glob.glob(face_image_dir))} images")

#         embeddings = np.asarray(embeddings).astype('float32')
#         faiss.normalize_L2(embeddings)
#         faiss_index.add(embeddings)
#         faiss.write_index(faiss_index, face_index_path)

#         np.savez(
#                 face_details_path, 
#                 user_names=user_names, 
#                 facial_areas=facial_areas, 
#                 face_confidences=face_confidences
#                 )
        
#     else:
#         faiss_index = faiss.read_index(face_index_path)
#         face_details = np.load(face_details_path)
#         user_names = face_details['user_names']
#         facial_areas = face_details['facial_areas']
#         face_confidences = face_details['face_confidences']

#     return faiss_index, user_names, facial_areas, face_confidences

# def extract_face_information_for_inference(img_path):
#     face_objs = DeepFace.represent(
#                                 img_path = img_path,
#                                 model_name = models[2],
#                                 enforce_detection = False
#                                 )
#     img_path = img_path.replace("\\", "/")

#     embeddings = []
#     facial_areas = []
#     face_confidences = []

#     if len(face_objs) == 0:
#         Warning(f"No faces detected in the image : {img_path}")
#     else:
#         for i in range(len(face_objs)):
#             embs = face_objs[i]['embedding']
#             facial_area = face_objs[i]['facial_area']
#             face_confidence = face_objs[i]['face_confidence']
#             x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

#             embeddings.append(embs)
#             facial_areas.append((x, y, w, h))   
#             face_confidences.append(face_confidence)

#     return embeddings, face_confidences, facial_areas

# def search_face_in_db(
#                     img_path, 
#                     face_index_path = 'models/face_index',
#                     face_details_path = 'models/face_details.npz',
#                     ):
#     index, user_names, _, _ = build_face_embedding_index(
#                                                         face_index_path = face_index_path,
#                                                         face_details_path = face_details_path,
#                                                         )
#     embeddings, face_confidences, facial_areas = extract_face_information_for_inference(img_path)

#     retrieved_user_names = []
#     retrieved_facial_areas = []
#     retrieved_face_confidences = []

#     if embeddings is not None:
#         for idx, emb in enumerate(embeddings):
#             if face_confidences[idx] >= 0.8:
#                 emb = np.array(emb).reshape(1, -1).astype('float32')
#                 faiss.normalize_L2(emb)
#                 D, I = index.search(emb, 5)
#                 I = np.array(I).squeeze()
#                 D = np.array(D).squeeze()
#                 user_name_list = [user_names[i] for i in I]
#                 user_name = max(set(user_name_list), key = user_name_list.count)
#                 avg_confidence = np.mean([d for i, d in zip(I, D) if user_names[i] == user_name])
#                 retrieved_face_confidences.append(np.round(avg_confidence, 3))
#                 retrieved_facial_areas.append(facial_areas[idx])
#                 retrieved_user_names.append(user_name)

#     return retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences

# def eculedian_distance(x1, y1, x2, y2):
#     return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# def face_image_inference(
#                         username,
#                         face_image_path
#                         ):
#     img = cv2.imread(face_image_path)
#     img_cp = img.copy()

#     img_cp, texts, face_centroids = head_pose_inference(img_cp, image_flag = True)

#     retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db(face_image_path)
#     for i in range(len(retrieved_user_names)):
#         x, y, w, h = retrieved_facial_areas[i]
#         face_centhroid_bbox = (x + w//2, y + h//2)
#         face_area = w * h
#         if (face_area >= 20000):
#             timestamp = datetime.now()
#             timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#             if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
#                 if (len(face_centroids) > 1) and (len(face_centroids) > 1):
#                     distances = [eculedian_distance(face_centhroid_bbox[0], face_centhroid_bbox[1], x, y) for x, y in face_centroids]
#                     head_pose_text = texts[np.argmin(distances)]
#                 else:
#                     head_pose_text = texts[0]
                    
#                 ffeatures_collection.insert_one({
#                                                 "exp_username": username,
#                                                 "det_username": retrieved_user_names[i],
#                                                 "head_pose": head_pose_text,
#                                                 "face_confidence": float(retrieved_face_confidences[i]),
#                                                 "timestamp": timestamp
#                                                 })
#                 det_username = retrieved_user_names[i]

#                 cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 font = cv.FONT_HERSHEY_SIMPLEX
#                 cv.putText(img_cp, f'User: {retrieved_user_names[i]}', (x-30, y-40), font, 1, (0, 255, 0), 2)
#             else:
#                 cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                 ffeatures_collection.insert_one({
#                                                 "exp_username": username,
#                                                 "det_username": "N/A",
#                                                 "head_pose": "Unknown",
#                                                 "face_confidence": "N/A",
#                                                 "timestamp": timestamp
#                                                 })
#                 det_username = "N/A"
                
#     return head_pose_text, det_username
#     # cv.imshow('Face Monitoring Inference', img_cp)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()

# def video_face_inference(
#                         username,
#                         is_vis = False
#                         ):
#     cap = cv.VideoCapture(0)
#     while cap.isOpened():
#         success, img = cap.read()
#         if not success:
#             break

#         img_cp = img.copy()
#         img_cp_ = cv2.flip(img_cp, 1)
#         cv.imwrite("data/temp_dir/temp.jpg", img_cp_)

#         img_cp, texts, face_centroids = head_pose_inference(img_cp)

#         retrieved_user_names, retrieved_facial_areas, retrieved_face_confidences = search_face_in_db("data/temp_dir/temp.jpg")
#         for i in range(len(retrieved_user_names)):
#             x, y, w, h = retrieved_facial_areas[i]
#             face_centhroid_bbox = (x + w//2, y + h//2)
#             face_area = w * h
#             if (face_area >= 20000):
#                 timestamp = datetime.now()
#                 timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#                 if (retrieved_user_names[i] == username) and (retrieved_face_confidences[i] >= 0.5):
#                     if (len(face_centroids) > 1) and (len(face_centroids) > 1):
#                         distances = [eculedian_distance(face_centhroid_bbox[0], face_centhroid_bbox[1], x, y) for x, y in face_centroids]
#                         head_pose_text = texts[np.argmin(distances)]
#                     elif len(face_centroids) == 1:
#                         head_pose_text = texts[0]

#                     else:
#                         head_pose_text = "UnKnown"

#                     ffeatures_collection.insert_one({
#                                                     "exp_username": username,
#                                                     "det_username": retrieved_user_names[i],
#                                                     "head_pose": head_pose_text,
#                                                     "face_confidence": float(retrieved_face_confidences[i]),
#                                                     "timestamp": timestamp
#                                                     })
                
#                     cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                     font = cv.FONT_HERSHEY_SIMPLEX
#                     cv.putText(img_cp, f'User: {retrieved_user_names[i]}', (x-30, y-40), font, 1, (0, 255, 0), 2)
#                 else:
#                     cv.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                     ffeatures_collection.insert_one({
#                                                     "exp_username": username,
#                                                     "det_username": "N/A",
#                                                     "head_pose": "Unknown",
#                                                     "face_confidence": "N/A",
#                                                     "timestamp": timestamp
#                                                     })
                
#         if is_vis:
#             cv.imshow('Face Monitoring Inference', img_cp)
#             if cv.waitKey(5) & 0xFF == 27:
#                 break

#     cap.release()
#     cv.destroyAllWindows()

# def face_analysis(
#                 username,
#                 x_min = 1440
#                 ):
#     current_time = datetime.now()
#     current_time_minus_x = current_time - timedelta(minutes=x_min)

#     current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
#     current_time_minus_x = current_time_minus_x.strftime("%Y-%m-%d %H:%M:%S")

#     data_user = ffeatures_collection.find({
#                                         "exp_username": username,
#                                         "timestamp": {
#                                                     "$gte": current_time_minus_x,
#                                                     "$lt": current_time
#                                                     }
#                                         })
#     data_user = pd.DataFrame(list(data_user))
    
#     if data_user.empty:
#         return None, None
    
#     else:
#         data_user = data_user.drop(columns = ["_id", "exp_username", "face_confidence", "timestamp"])
#         n_forward = len(data_user[data_user["head_pose"] == "Forward"])
#         n_detected = len(data_user[data_user["det_username"] == username])
#         n_total = len(data_user)

#         detected_percentage = (n_detected/n_total)*100
#         forward_percentage = (n_forward/n_total)*100

#         detected_percentage = round(detected_percentage, 2)
#         forward_percentage = round(forward_percentage, 2)


#         detected_percentage = f"{detected_percentage} %"
#         forward_percentage = f"{forward_percentage} %"

#         return {
#                 "detected_percentage": detected_percentage,
#                 "forward_percentage": forward_percentage
#                 }
    
# # # face_image_inference("Isuru Alagiyawanna", 'data/facedb/Isuru Alagiyawanna/IMG-20240804-WA0009.jpg')
# # face_image_inference("Akshay Kumar", 'data/test_images/qqq.jpg')
# # video_face_inference("Isuru Alagiyawanna", is_vis=True)


import cv2
import time # cv2 එක්ක cv තිබ්බා, time අලුතෙන් දැම්මා
import numpy as np
import pandas as pd
import yaml
import pymongo
import mediapipe as mp
import faiss
import glob
import os
from deepface import DeepFace
from datetime import datetime, timedelta

# --- Configuration and Global Variables ---
CONFIG_FILE = 'secrets.yaml' # secrets.yaml තියෙන තැනට path එක හරියට දෙන්න
MODELS_DIR = 'models'
FAISS_INDEX_FILE = os.path.join(MODELS_DIR, 'face_index')
FAISS_DETAILS_FILE = os.path.join(MODELS_DIR, 'face_details.npz')
FAISS_DIMENSION = 512  # For Facenet512
DEEPFACE_MODEL_NAME = "Facenet512" # කලින් models[2] පාවිච්චි කළා, දැන් කෙලින්ම නම දාමු

# Global FAISS index and related data
GLOBAL_FAISS_INDEX = None
GLOBAL_DB_USER_NAMES = None
GLOBAL_DB_FACIAL_AREAS = None
GLOBAL_DB_FACE_CONFIDENCES = None

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('data/temp_dir', exist_ok=True) # For temp images

# --- MongoDB Connection ---
MONGO_DB_URI_ENV = "MONGO_DB_URI" # Environment variable name
db_client = None
ffeatures_collection = None

try:
    # Load secrets only if the file exists
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            secrets = yaml.load(f, Loader=yaml.FullLoader)
        if MONGO_DB_URI_ENV not in os.environ and 'MONGO_DB_URI' in secrets: # Set env var if not already set
            os.environ[MONGO_DB_URI_ENV] = secrets['MONGO_DB_URI']

    if MONGO_DB_URI_ENV in os.environ:
        db_client = pymongo.MongoClient(os.environ[MONGO_DB_URI_ENV])
        db = db_client['Elearning'] # DB නම
        ffeatures_collection = db['ffeatures'] # Collection නම
        print("Successfully connected to MongoDB.")
    else:
        print("MongoDB URI not found in environment variables or secrets.yaml. MongoDB features will be disabled.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db_client = None # Ensure client is None on error
    ffeatures_collection = None
# --- End MongoDB Connection ---


# --- MediaPipe Face Mesh Setup ---
try:
    face_mesh_mp_solution = mp.solutions.face_mesh
    face_mesh = face_mesh_mp_solution.FaceMesh(
        max_num_faces=1, # Process one face for head pose for simplicity
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), circle_radius=1, thickness=1) # Thickness අඩු කළා
except Exception as e:
    print(f"Error initializing MediaPipe Face Mesh: {e}")
    face_mesh = None
# --- End MediaPipe Face Mesh Setup ---


# --- DeepFace Models List (Reference Only) ---
# models_ref = [
# "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
# "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
# ]
# --- End DeepFace Models List ---


def head_pose_inference(image_input, image_is_file_path=False): # Parameter name changed for clarity
    if face_mesh is None:
        print("MediaPipe Face Mesh not initialized. Skipping head pose.")
        return image_input, ["MediaPipe Error"], [] # Return original image and error state

    start_time = time.time() # Renamed variable
    
    # Read image if path is given
    if image_is_file_path:
        image_cv = cv2.imread(image_input)
        if image_cv is None:
            print(f"Error: Could not read image from path: {image_input}")
            return None, ["Image Read Error"], []
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    else: # Assuming image_input is already a NumPy array (e.g., from webcam)
        image_cv = cv2.cvtColor(cv2.flip(image_input, 1), cv2.COLOR_BGR2RGB) # Flip for webcam-like view

    image_cv.flags.writeable = False
    results = face_mesh.process(image_cv)
    image_cv.flags.writeable = True
    image_cv_output = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR) # Convert back for display/saving

    img_h, img_w, _ = image_cv_output.shape # Use output image shape
    all_texts = []
    all_face_centroids = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d_coords = [] # Renamed variable
            face_3d_coords = [] # Renamed variable

            # Selected landmark indices for head pose estimation
            landmark_indices = [33, 263, 1, 61, 291, 199] # Nose, L/R eye, L/R mouth, chin
            nose_2d, nose_3d = None, None

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in landmark_indices:
                    if idx == 1:  # Nose tip
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000) # Z scaling factor
                    
                    x_coord, y_coord = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d_coords.append([x_coord, y_coord])
                    face_3d_coords.append([x_coord, y_coord, lm.z * img_w]) # lm.z is normalized; scale by img_w or a constant

            if not face_2d_coords or not nose_2d: # Check if critical landmarks were found
                all_texts.append("Landmarks Error")
                all_face_centroids.append((img_w // 2, img_h // 2)) # Default centroid
                continue

            face_2d_np = np.array(face_2d_coords, dtype=np.float64)
            face_3d_np = np.array(face_3d_coords, dtype=np.float64)
            
            current_face_centroid = np.mean(face_2d_np, axis=0)
            all_face_centroids.append(current_face_centroid)

            focal_length_val = 1 * img_w # Renamed variable
            cam_matrix = np.array([
                [focal_length_val, 0, img_w / 2],
                [0, focal_length_val, img_h / 2], # Used img_h here
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64) # Renamed variable

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d_np, face_2d_np, cam_matrix, dist_coeffs
            ) # Renamed variables

            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec) # Renamed variable
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                
                # Convert angles to degrees, adjust for intuitive interpretation
                pitch_angle = angles[0]  # X-axis rotation
                yaw_angle = angles[1]    # Y-axis rotation
                # roll_angle = angles[2]   # Z-axis rotation (not used for L/R/U/D text)

                text_direction = "Forward"
                if yaw_angle < -10: text_direction = "Looking Left"
                elif yaw_angle > 10: text_direction = "Looking Right"
                elif pitch_angle < -10: text_direction = "Looking Down" # Pitch for Up/Down
                elif pitch_angle > 10: text_direction = "Looking Up"
                all_texts.append(text_direction)

                # Visualization (optional, can be removed for pure inference)
                # nose_3d_projection, _ = cv2.projectPoints(
                #     np.array([nose_3d]), rot_vec, trans_vec, cam_matrix, dist_coeffs
                # )
                # p1_viz = (int(nose_2d[0]), int(nose_2d[1]))
                # p2_viz = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                # cv2.line(image_cv_output, p1_viz, p2_viz, (255, 0, 0), 2)
                # cv2.putText(image_cv_output, text_direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                all_texts.append("SolvePnP Failed")
            
            # Draw landmarks (optional)
            # mp_drawing.draw_landmarks(
            #     image=image_cv_output, landmark_list=face_landmarks,
            #     connections=face_mesh_mp_solution.FACEMESH_CONTOURS, # Corrected constant
            #     landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec
            # )
    else:
        all_texts.append("No Face Detected")
        # Return a default centroid if no face is detected
        all_face_centroids.append((img_w // 2, img_h // 2))


    # fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
    # cv2.putText(image_cv_output, f'FPS: {int(fps)}', (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return image_cv_output, all_texts, all_face_centroids


def extract_face_information_for_db(img_path_db): # Renamed param
    # This function extracts info for ONE image, expecting ONE clear face for DB.
    try:
        # DeepFace.represent returns a list of dictionaries, one for each detected face.
        face_data_list = DeepFace.represent(
            img_path=img_path_db,
            model_name=DEEPFACE_MODEL_NAME, # Use global model name
            enforce_detection=False, # Allow processing if detection is weak/multiple
            detector_backend='opencv' # or 'skip' if images are pre-cropped
        )

        img_path_norm = img_path_db.replace("\\", "/")
        # Assuming path structure like 'data/facedb/username/image.jpg'
        path_parts = img_path_norm.split("/")
        user_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_user"

        if not face_data_list: # No faces detected or error
            print(f"Warning (DB): No face data from DeepFace.represent for {img_path_db}")
            return None, None, None, None
        
        # For DB, we typically want one primary face. If multiple, take the first.
        # More advanced: choose largest face or one with highest confidence if available.
        if len(face_data_list) > 1:
            print(f"Warning (DB): Multiple faces ({len(face_data_list)}) in {img_path_db}. Using first one.")
        
        target_face_data = face_data_list[0] # Get the first (or primary) face's data

        embeddings = target_face_data.get('embedding')
        facial_area_dict = target_face_data.get('facial_area')
        # Confidence might not always be present or named 'face_confidence'
        face_confidence = target_face_data.get('face_confidence', target_face_data.get('confidence', 1.0))


        if embeddings is None or len(embeddings) != FAISS_DIMENSION:
            print(f"Error (DB): Embedding issue for {img_path_db}. Dim: {len(embeddings) if embeddings else 'None'}")
            return None, None, None, None
        if facial_area_dict is None:
            print(f"Error (DB): Facial area not found for {img_path_db}")
            return None, None, None, None

        x = facial_area_dict.get('x')
        y = facial_area_dict.get('y')
        w = facial_area_dict.get('w')
        h = facial_area_dict.get('h')

        if any(v is None for v in [x, y, w, h]): # Check if any coordinate is None
            print(f"Error (DB): Incomplete facial area data for {img_path_db}")
            return None, None, None, None
            
        return embeddings, float(face_confidence), (x, y, w, h), user_name

    except Exception as e:
        print(f"DeepFace Error in extract_face_information_for_db for {img_path_db}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def build_face_embedding_index(
    d_param=FAISS_DIMENSION, # Renamed param
    index_file_path=FAISS_INDEX_FILE, # Renamed param
    images_dir=os.path.join('data', 'facedb', '*', '*.jpg'), # Default to all users, all images
    details_file_path=FAISS_DETAILS_FILE, # Renamed param
    force_rebuild_flag=False # Renamed param
):
    global GLOBAL_FAISS_INDEX, GLOBAL_DB_USER_NAMES, GLOBAL_DB_FACIAL_AREAS, GLOBAL_DB_FACE_CONFIDENCES

    if force_rebuild_flag:
        if os.path.exists(index_file_path): os.remove(index_file_path)
        if os.path.exists(details_file_path): os.remove(details_file_path)
        print("Force rebuild: Existing index and details files removed.")

    if not os.path.exists(index_file_path) or not os.path.exists(details_file_path):
        print(f"Building new FAISS index (dim={d_param}) and details file...")
        # For Facenet (L2 normalized embeddings), IndexFlatL2 is common.
        # METRIC_INNER_PRODUCT also works for normalized vectors.
        current_faiss_index = faiss.IndexFlatL2(d_param)
        # current_faiss_index = faiss.index_factory(d_param, "Flat", faiss.METRIC_INNER_PRODUCT)


        all_embeddings = []
        all_user_names = []
        all_facial_areas = []
        all_face_confidences = []
        
        image_paths_list = list(glob.glob(images_dir)) # Renamed variable
        total_images_count = len(image_paths_list) # Renamed variable
        print(f"Found {total_images_count} images in '{images_dir}' for index build.")

        for i, img_path_build in enumerate(image_paths_list): # Renamed param
            emb, face_conf, facial_area, user_name = extract_face_information_for_db(img_path_build)
            if emb is not None: # emb is a list of floats
                all_embeddings.append(emb)
                all_user_names.append(user_name)
                all_facial_areas.append(facial_area) # facial_area is a tuple (x,y,w,h)
                all_face_confidences.append(face_conf)

            if (i + 1) % 50 == 0 or (i + 1) == total_images_count:
                print(f"Processed {i+1}/{total_images_count} images for index building...")

        if not all_embeddings:
            print("No embeddings extracted. FAISS index will be empty but saved.")
            # Even if empty, save a validly dimensioned index to avoid errors on load
            # IndexFlatL2 does not require training if d is set.
            # if not current_faiss_index.is_trained: current_faiss_index.train(np.zeros((0, d_param), dtype='float32'))
            faiss.write_index(current_faiss_index, index_file_path)
            np.savez(
                details_file_path,
                user_names=np.array(all_user_names, dtype=str), # Ensure dtype for empty arrays
                facial_areas=np.array(all_facial_areas, dtype=object),
                face_confidences=np.array(all_face_confidences, dtype=float)
            )
        else:
            embeddings_np = np.asarray(all_embeddings, dtype='float32')
            if embeddings_np.ndim == 1 and embeddings_np.shape[0] == d_param: # Single embedding case
                 embeddings_np = embeddings_np.reshape(1, -1)
            
            if embeddings_np.shape[0] > 0 and embeddings_np.shape[1] != d_param: # Check dim only if there are embeddings
                print(f"CRITICAL ERROR: Embeddings dim ({embeddings_np.shape[1]}) != index dim ({d_param}). Index NOT built.")
                return None, np.array([]), np.array([]), np.array([])
            
            print(f"Shape of embeddings array for FAISS: {embeddings_np.shape}")
            faiss.normalize_L2(embeddings_np) # Normalize for L2 distance / Inner Product similarity
            current_faiss_index.add(embeddings_np)
            faiss.write_index(current_faiss_index, index_file_path)

            np.savez(
                details_file_path,
                user_names=np.array(all_user_names, dtype=str),
                facial_areas=np.array(all_facial_areas, dtype=object), # Store tuples as objects
                face_confidences=np.array(all_face_confidences, dtype=float)
            )
            print(f"Successfully built & saved FAISS index ({current_faiss_index.ntotal} vectors) and details.")
        
        # Update global variables after building
        GLOBAL_FAISS_INDEX = current_faiss_index
        GLOBAL_DB_USER_NAMES = np.array(all_user_names, dtype=str)
        GLOBAL_DB_FACIAL_AREAS = np.array(all_facial_areas, dtype=object)
        GLOBAL_DB_FACE_CONFIDENCES = np.array(all_face_confidences, dtype=float)
        
    else: # Index files exist, load them
        print(f"Loading existing FAISS index from {index_file_path}...")
        try:
            GLOBAL_FAISS_INDEX = faiss.read_index(index_file_path)
            if GLOBAL_FAISS_INDEX.d != d_param:
                print(f"WARNING: Loaded FAISS index dim ({GLOBAL_FAISS_INDEX.d}) != expected dim ({d_param}).")
                print("Consider rebuilding with force_rebuild_flag=True.")
                # Optionally, could trigger a rebuild or return None to indicate error
            
            details_data = np.load(details_file_path, allow_pickle=True) # allow_pickle for object arrays
            GLOBAL_DB_USER_NAMES = details_data['user_names']
            GLOBAL_DB_FACIAL_AREAS = details_data['facial_areas']
            GLOBAL_DB_FACE_CONFIDENCES = details_data['face_confidences']
            print(f"Loaded FAISS index ({GLOBAL_FAISS_INDEX.ntotal} vectors, dim={GLOBAL_FAISS_INDEX.d}) and details.")
        except Exception as e:
            print(f"Error loading existing FAISS index or details: {e}")
            print("Attempting to rebuild the index...")
            return build_face_embedding_index(d_param, index_file_path, images_dir, details_file_path, force_rebuild_flag=True)


    return GLOBAL_FAISS_INDEX, GLOBAL_DB_USER_NAMES, GLOBAL_DB_FACIAL_AREAS, GLOBAL_DB_FACE_CONFIDENCES

# --- Function to initialize/load the global FAISS index (called once at app startup) ---
def initialize_faiss_system(force_build=False):
    print("Initializing FAISS system...")
    # This will build if not exists, or load if exists.
    # Pass the default image directory to build from all users if index is new.
    build_face_embedding_index(
        d_param=FAISS_DIMENSION,
        images_dir=os.path.join('data', 'facedb', '*', '*.jpg'), # Build from all images if new
        force_rebuild_flag=force_build
    )
    if GLOBAL_FAISS_INDEX and GLOBAL_FAISS_INDEX.ntotal > 0:
        print(f"FAISS system initialized. Index vectors: {GLOBAL_FAISS_INDEX.ntotal}, Dim: {GLOBAL_FAISS_INDEX.d}")
    elif GLOBAL_FAISS_INDEX:
        print("FAISS system initialized, but index is empty.")
    else:
        print("FAISS system initialization failed or index is empty and could not be built.")

# --- Call initialization at module load (e.g., when Flask app imports this) ---
# initialize_faiss_system(force_build=False) # Set force_build=True if you always want to rebuild on start
# Note: It's often better to call this explicitly from app.py after Flask app is created.
# For now, if inference.py is run as a script, it will try to initialize.


def extract_face_information_for_inference(img_path_inf): # Renamed param
    # Extracts info for ALL faces in a query image.
    all_query_embeddings = [] # Renamed variable
    all_query_facial_areas = [] # Renamed variable
    all_query_face_confidences = [] # Renamed variable
    try:
        face_data_list_inf = DeepFace.represent( # Renamed variable
            img_path=img_path_inf,
            model_name=DEEPFACE_MODEL_NAME,
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if not face_data_list_inf:
            # print(f"Warning (Inference): No faces detected in {img_path_inf}") # Can be noisy
            pass
        else:
            for face_data in face_data_list_inf:
                emb_inf = face_data.get('embedding') # Renamed variable
                if emb_inf and len(emb_inf) == FAISS_DIMENSION:
                    all_query_embeddings.append(emb_inf)
                    
                    facial_area_dict_inf = face_data['facial_area'] # Renamed variable
                    conf_inf = face_data.get('face_confidence', face_data.get('confidence', 1.0)) # Renamed variable
                    x_inf, y_inf, w_inf, h_inf = facial_area_dict_inf['x'], facial_area_dict_inf['y'], facial_area_dict_inf['w'], facial_area_dict_inf['h']
                    
                    all_query_facial_areas.append((x_inf, y_inf, w_inf, h_inf))
                    all_query_face_confidences.append(conf_inf)
                # else:
                    # print(f"Warning (Inference): Skipping face in {img_path_inf} due to embedding issue.")
        
        return all_query_embeddings, all_query_face_confidences, all_query_facial_areas
    except Exception as e:
        print(f"DeepFace Error in extract_face_information_for_inference for {img_path_inf}: {e}")
        return [], [], []


def search_face_in_db(
    query_img_path, # Renamed param
    num_neighbors_k=5, # Renamed param
    similarity_thresh=0.45 # Renamed param (L2 dist based similarity)
):
    # This function now uses the GLOBAL_FAISS_INDEX
    if GLOBAL_FAISS_INDEX is None or GLOBAL_DB_USER_NAMES is None:
        print("Search Error: Global FAISS index or user names not loaded. Trying to initialize...")
        initialize_faiss_system() # Try to load/build it
        if GLOBAL_FAISS_INDEX is None or GLOBAL_DB_USER_NAMES is None:
            print("Search Error: FAISS system could not be initialized.")
            return [], [], [] # Return empty if still not loaded
            
    if GLOBAL_FAISS_INDEX.ntotal == 0:
        # print("Search Info: FAISS index is empty.") # Can be noisy
        return [], [], []

    # Extract embeddings from the query image
    query_embeddings_list, _, query_facial_areas_list = extract_face_information_for_inference(query_img_path)

    all_retrieved_users = []
    all_retrieved_areas_output = [] # Renamed variable
    all_retrieved_scores_output = [] # Renamed variable

    if not query_embeddings_list:
        # print(f"No valid faces/embeddings found in query image {query_img_path} for search.") # Can be noisy
        return [], [], []

    for i in range(len(query_embeddings_list)):
        current_query_embedding = np.array(query_embeddings_list[i], dtype='float32').reshape(1, -1)
        faiss.normalize_L2(current_query_embedding) # Normalize query embedding

        # D: squared L2 distances, I: indices of neighbors
        distances_l2_sq, neighbor_indices = GLOBAL_FAISS_INDEX.search(current_query_embedding, num_neighbors_k)
        
        best_match_user_for_face = "Unknown"
        highest_similarity_for_face = 0.0 # Using 0 as a baseline

        # Process neighbors for the current query face
        for j, db_idx in enumerate(neighbor_indices[0]): # For the first (and only) query embedding
            if 0 <= db_idx < len(GLOBAL_DB_USER_NAMES): # Check for valid index
                dist_sq_val = distances_l2_sq[0][j]
                # Cosine similarity from L2 distance: sim = 1 - (L2_dist^2 / 2)
                # Max L2_dist^2 for normalized vectors is 4 (when vectors are opposite).
                # So, similarity ranges from -1 to 1.
                current_similarity = 1.0 - (dist_sq_val / 2.0) if dist_sq_val <= 4.0 else -1.0 
                
                # print(f"  QueryFace{i}-Neighbor{j}: DB_idx={db_idx}, User='{GLOBAL_DB_USER_NAMES[db_idx]}', L2^2={dist_sq_val:.4f}, Sim={current_similarity:.4f}")

                if current_similarity > highest_similarity_for_face and current_similarity >= similarity_thresh:
                    highest_similarity_for_face = current_similarity
                    best_match_user_for_face = GLOBAL_DB_USER_NAMES[db_idx]
            # else:
                # print(f"  QueryFace{i}-Neighbor{j}: Invalid DB index {db_idx} from FAISS.")
        
        all_retrieved_users.append(best_match_user_for_face)
        all_retrieved_areas_output.append(query_facial_areas_list[i]) # Facial area of the query face
        all_retrieved_scores_output.append(np.round(highest_similarity_for_face, 3) if best_match_user_for_face != "Unknown" else 0.0)
        
        # if best_match_user_for_face != "Unknown":
        #     print(f"Query face {i+1} in {query_img_path} matched: {best_match_user_for_face} (Sim: {highest_similarity_for_face:.3f})")
        # else:
        #     print(f"Query face {i+1} in {query_img_path}: No strong match found.")

    return all_retrieved_users, all_retrieved_areas_output, all_retrieved_scores_output


def eculedian_distance(x1_coord, y1_coord, x2_coord, y2_coord): # Renamed params
    return np.sqrt((x1_coord - x2_coord)**2 + (y1_coord - y2_coord)**2)


def face_image_inference(expected_username, image_file_path): # Renamed params
    # This is the main function called by your Flask endpoint for single image processing.
    # print(f"--- face_image_inference: User='{expected_username}', Image='{image_file_path}' ---")

    # 1. Head Pose (Optional visualization can be done by caller if needed)
    _, head_pose_texts_list, face_centroids_list = head_pose_inference(image_file_path, image_is_file_path=True)
    
    # Assuming one primary head pose if multiple detected, or "Unknown"
    primary_head_pose = head_pose_texts_list[0] if head_pose_texts_list else "Head Pose N/A"

    # 2. Face Recognition (Search in DB)
    # search_face_in_db now uses global index
    retrieved_db_users, retrieved_db_areas, retrieved_db_scores = search_face_in_db(image_file_path)

    detected_username_from_db = "Unknown" # Default
    final_confidence_score = 0.0

    # Process recognition results:
    # If multiple faces detected in query image, decide which one to use for verification.
    # For simplicity, if the expected_username is found among retrieved_db_users with good score,
    # we consider that the primary result.
    if retrieved_db_users: # If any faces were recognized
        best_score_for_expected_user = 0
        matched_idx_for_expected_user = -1

        for i in range(len(retrieved_db_users)):
            if retrieved_db_users[i] == expected_username and retrieved_db_scores[i] > best_score_for_expected_user:
                best_score_for_expected_user = retrieved_db_scores[i]
                matched_idx_for_expected_user = i
        
        if matched_idx_for_expected_user != -1: # Expected user found with some confidence
            detected_username_from_db = expected_username
            final_confidence_score = retrieved_db_scores[matched_idx_for_expected_user]
            # If expected user is found, try to match head pose to this specific face
            if face_centroids_list and retrieved_db_areas:
                query_face_bbox = retrieved_db_areas[matched_idx_for_expected_user] # (x,y,w,h)
                query_face_centroid = (query_face_bbox[0] + query_face_bbox[2]//2, query_face_bbox[1] + query_face_bbox[3]//2)
                
                if head_pose_texts_list and len(head_pose_texts_list) == len(face_centroids_list):
                    min_dist = float('inf')
                    closest_head_pose_idx = -1
                    for k, hp_centroid in enumerate(face_centroids_list):
                        dist = eculedian_distance(query_face_centroid[0], query_face_centroid[1], hp_centroid[0], hp_centroid[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_head_pose_idx = k
                    if closest_head_pose_idx != -1:
                        primary_head_pose = head_pose_texts_list[closest_head_pose_idx]
        
        elif retrieved_db_users[0] != "Unknown": # If expected user not found, report the best other known match
            detected_username_from_db = retrieved_db_users[0] # Best match if not expected
            final_confidence_score = retrieved_db_scores[0]
            # primary_head_pose would be the general one if not matched to a specific recognized face
        
    # 3. Log to MongoDB (if connected)
    if ffeatures_collection is not None:
        timestamp_now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Renamed
        log_entry = {
            "exp_username": expected_username,
            "det_username": detected_username_from_db, # This is the recognized user
            "head_pose": primary_head_pose,
            "face_confidence": float(final_confidence_score), # Similarity score
            "timestamp": timestamp_now_str
        }
        try:
            ffeatures_collection.insert_one(log_entry)
            # print(f"Logged to MongoDB: {log_entry}")
        except Exception as e_mongo:
            print(f"MongoDB insert error: {e_mongo}")
    
    # print(f"--- face_image_inference result: HeadPose='{primary_head_pose}', DetectedUser='{detected_username_from_db}' ---")
    return primary_head_pose, detected_username_from_db


def video_face_inference(expected_username_video, visualize_output=False): # Renamed params
    # This function is for continuous webcam processing.
    if GLOBAL_FAISS_INDEX is None: # Ensure index is loaded
        print("Video Inference Error: FAISS index not loaded. Please initialize system.")
        return

    cap = cv2.VideoCapture(0) # Default webcam
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    temp_img_for_video = os.path.join('data', 'temp_dir', 'temp_video_frame.jpg') # Renamed

    print("Starting video face inference. Press 'ESC' or 'q' in window to quit.")
    while cap.isOpened():
        success, frame_orig = cap.read() # Renamed
        if not success:
            print("Webcam frame capture failed.")
            break

        # Process a copy for DeepFace/MediaPipe
        frame_to_process = frame_orig.copy()
        # Flipping is done by head_pose_inference if image_is_file_path=False
        # For DeepFace, we save the non-flipped or correctly oriented frame
        cv2.imwrite(temp_img_for_video, cv2.flip(frame_to_process, 1)) # Save flipped for consistency if DeepFace expects that

        # Head Pose
        frame_with_hp, hp_texts, hp_centroids = head_pose_inference(frame_to_process, image_is_file_path=False) # Pass raw frame

        # Face Recognition
        video_ret_users, video_ret_areas, video_ret_scores = search_face_in_db(temp_img_for_video)

        # Combine and Display/Log
        for i in range(len(video_ret_users)):
            x_v, y_v, w_v, h_v = video_ret_areas[i] # Renamed variables
            rec_user = video_ret_users[i]
            rec_score = video_ret_scores[i]
            
            current_loop_head_pose = "Unknown HP" # Default head pose for this face
            # Match recognized face to closest head pose centroid
            if hp_centroids and hp_texts and len(hp_texts) == len(hp_centroids):
                rec_face_centroid = (x_v + w_v//2, y_v + h_v//2)
                min_dist_hp = float('inf')
                closest_hp_idx = -1
                for k_hp, centroid_hp in enumerate(hp_centroids):
                    dist_val = eculedian_distance(rec_face_centroid[0], rec_face_centroid[1], centroid_hp[0], centroid_hp[1])
                    if dist_val < min_dist_hp:
                        min_dist_hp = dist_val
                        closest_hp_idx = k_hp
                if closest_hp_idx != -1:
                    current_loop_head_pose = hp_texts[closest_hp_idx]
            
            color_rect = (0, 0, 255) # Red for unknown/mismatch
            if rec_user == expected_username_video and rec_score >= 0.5: # Verification threshold
                color_rect = (0, 255, 0) # Green for match
            
            # Log to MongoDB
            if ffeatures_collection is not None:
                ts_video = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry_video = {
                    "exp_username": expected_username_video,
                    "det_username": rec_user,
                    "head_pose": current_loop_head_pose,
                    "face_confidence": float(rec_score),
                    "timestamp": ts_video
                }
                try:
                    ffeatures_collection.insert_one(log_entry_video)
                except Exception as e_mongo_vid:
                    print(f"MongoDB (video) insert error: {e_mongo_vid}")

            # Visualization on the frame used for headpose (which was flipped)
            if visualize_output:
                cv2.rectangle(frame_with_hp, (x_v, y_v), (x_v + w_v, y_v + h_v), color_rect, 2)
                label_text = f"{rec_user} ({rec_score:.2f}) HP: {current_loop_head_pose}"
                cv2.putText(frame_with_hp, label_text, (x_v, y_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rect, 2)
        
        if visualize_output:
            cv2.imshow('Video Face Inference (Press ESC or q to quit)', frame_with_hp) # Display the annotated frame
            key_pressed = cv2.waitKey(5) & 0xFF
            if key_pressed == 27 or key_pressed == ord('q'): # ESC or 'q'
                break
        else: # If not visualizing, add a small delay or condition to break loop for non-GUI use
            time.sleep(0.1) # Example delay
            # Add a counter or time limit if running non-interactively

    cap.release()
    if visualize_output: cv2.destroyAllWindows()
    if os.path.exists(temp_img_for_video): os.remove(temp_img_for_video) # Clean up
    print("Video face inference stopped.")


def face_analysis(username_analyze, minutes_back=1440): # Renamed params
    if ffeatures_collection is None:
        print("Face Analysis Error: MongoDB not connected.")
        return {"error": "Database not available"}, None # Return error structure

    time_now = datetime.now() # Renamed
    time_past = time_now - timedelta(minutes=minutes_back) # Renamed

    # Format for MongoDB query
    # query_time_now_str = time_now.strftime("%Y-%m-%d %H:%M:%S") # Not used in query directly
    query_time_past_str = time_past.strftime("%Y-%m-%d %H:%M:%S")

    # Find documents for the user within the time window
    user_data_cursor = ffeatures_collection.find({ # Renamed
        "exp_username": username_analyze,
        "timestamp": {"$gte": query_time_past_str} # Only lower bound needed if upper is current time implicitly
    })
    user_data_df = pd.DataFrame(list(user_data_cursor)) # Renamed

    if user_data_df.empty:
        print(f"No data found for user '{username_analyze}' in the last {minutes_back} minutes.")
        # Return structure consistent with expected success, but with "N/A" or 0 values
        return {
            "detected_percentage": "0.00 %",
            "forward_percentage": "0.00 %",
            "total_records": 0,
            "message": "No data found"
        } # Removed second None return

    # Drop unnecessary columns for analysis
    # data_user_analysis = user_data_df.drop(columns=["_id", "exp_username", "face_confidence", "timestamp"], errors='ignore')
    # We need 'det_username' and 'head_pose'
    
    n_total_records = len(user_data_df)
    n_forward_pose = len(user_data_df[user_data_df["head_pose"] == "Forward"])
    n_user_detected = len(user_data_df[user_data_df["det_username"] == username_analyze])

    detected_perc = (n_user_detected / n_total_records) * 100 if n_total_records > 0 else 0
    forward_perc = (n_forward_pose / n_total_records) * 100 if n_total_records > 0 else 0

    return { # Return a dictionary directly
        "detected_percentage": f"{detected_perc:.2f} %",
        "forward_percentage": f"{forward_perc:.2f} %",
        "total_records": n_total_records,
        "target_user": username_analyze,
        "time_window_minutes": minutes_back
    } # Removed second None return


# --- Example Usage (for testing inference.py directly) ---
if __name__ == '__main__':
    print("Running inference.py as main script for testing...")
    
    # STEP 1: Initialize FAISS system (Builds index if not present, or loads it)
    # Set force_build=True if you want to rebuild the index from scratch every time.
    # For first run after deleting old index files, this will build it.
    initialize_faiss_system(force_build=False) # Important to run this!

    if GLOBAL_FAISS_INDEX and GLOBAL_FAISS_INDEX.ntotal > 0:
        print(f"\n--- Testing face_image_inference ---")
        test_user = "Isuru Alagiyawanna" # Change to a user in your facedb
        # Ensure this image path is correct and the user 'Isuru Alagiyawanna' exists in 'data/facedb/'
        # Example: 'data/facedb/Isuru Alagiyawanna/some_image.jpg'
        test_image = 'data/test_images/purna (6).jpg' # Use a general test image

        if os.path.exists(test_image):
            # If test_user is not in DB, det_username will be "Unknown" or another user
            # If test_user IS in DB, det_username should ideally be test_user
            head_p, det_u = face_image_inference(test_user, test_image)
            print(f"Image Inference for '{test_user}' in '{os.path.basename(test_image)}':")
            print(f"  Detected User: {det_u}, Head Pose: {head_p}")
            
            # Test with a known user from your facedb
            # Assuming 'purna' is a user in your 'data/facedb/purna/' directory
            known_user_in_db = "purna"
            # Use an image of 'purna'
            purna_image_path = 'data/facedb/purna/purna (10).jpg' # Replace with an actual image of purna
            if os.path.exists(purna_image_path):
                 head_p_known, det_u_known = face_image_inference(known_user_in_db, purna_image_path)
                 print(f"Image Inference for known user '{known_user_in_db}' in '{os.path.basename(purna_image_path)}':")
                 print(f"  Detected User: {det_u_known}, Head Pose: {head_p_known}")
            else:
                print(f"Image for known user '{known_user_in_db}' not found: {purna_image_path}")

        else:
            print(f"Test image not found: {test_image}")

        # print(f"\n--- Testing face_analysis ---")
        # analysis_results = face_analysis(test_user, minutes_back=60)
        # print(f"Face Analysis for '{test_user}': {analysis_results}")

        # print(f"\n--- Testing video_face_inference (Press ESC or q in window to quit) ---")
        # video_face_inference(test_user, visualize_output=True)

    else:
        print("FAISS index not loaded or empty. Cannot run tests.")
    
    print("\n--- End of inference.py tests ---")