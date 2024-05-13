# import cv2
# import mediapipe as mp
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic
#
# def create_frame_landmark_df(results,frame,xyz):
#   xyz_skel = xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True) \
#             .copy()
#   face = pd.DataFrame()
#   pose = pd.DataFrame()
#   left_hand = pd.DataFrame()
#   right_hand = pd.DataFrame()
#
#   if results.face_landmarks:
#     for i,point in enumerate(results.face_landmarks.landmark):
#       face.loc[i,['x','y','z']] = [point.x, point.y,point.z]
#   if results.pose_landmarks:
#     for i,point in enumerate(results.pose_landmarks.landmark):
#       pose.loc[i,['x','y','z']] = [point.x, point.y,point.z]
#   if results.left_hand_landmarks:
#     for i,point in enumerate(results.left_hand_landmarks.landmark):
#       left_hand.loc[i,['x','y','z']] = [point.x, point.y,point.z]
#   if results.right_hand_landmarks:
#     for i,point in enumerate(results.right_hand_landmarks.landmark):
#       right_hand.loc[i,['x','y','z']] = [point.x, point.y, point.z]
#   face = face.reset_index()\
#       .rename(columns={'index':'landmark_index'})\
#       .assign(type='face')
#   pose = pose.reset_index() \
#     .rename(columns={'index': 'landmark_index'}) \
#     .assign(type='pose')
#   left_hand = left_hand.reset_index() \
#     .rename(columns={'index': 'landmark_index'}) \
#     .assign(type='left_hand')
#   right_hand = right_hand.reset_index() \
#     .rename(columns={'index': 'landmark_index'}) \
#     .assign(type='right_hand')
#   landmarks = pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True)
#   landmarks = xyz_skel.merge(landmarks,on=['type','landmark_index'], how='left')
#   landmarks = landmarks.assign(frame=frame)
#   return landmarks
# def do_capture_loop(xyz):
#   all_landmarks =[]
#   # For webcam input:
#   cap = cv2.VideoCapture(0)
#   with mp_holistic.Holistic(
#       min_detection_confidence=0.5,
#       min_tracking_confidence=0.5) as holistic:
#     frame = 0
#     while cap.isOpened():
#       frame +=1
#       success, image = cap.read()
#       if not success:
#         print("Ignoring empty camera frame.")
#         # If loading a video, use 'break' instead of 'continue'.
#         continue
#
#       # To improve performance, optionally mark the image as not writeable to
#       # pass by reference.
#       image.flags.writeable = False
#       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#       results = holistic.process(image)
#
#       landmarks = create_frame_landmark_df(results,frame,xyz)
#       all_landmarks.append(landmarks)
#
#       image.flags.writeable = True
#       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#       mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_contours_style())
#       mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles
#         .get_default_pose_landmarks_style())
#       # Flip the image horizontally for a selfie-view display.
#       cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
#       if cv2.waitKey(5) & 0xFF == 27:
#         break
#   return all_landmarks
#
# if __name__ == "__main__":
#   pq_file = '10042041.parquet'
#   xyz = pd.read_parquet(pq_file)
#   landmarks = do_capture_loop(xyz)
#   landmarks = pd.concat(landmarks).reset_index(drop=True).to_parquet('output.parquet')

import pandas as pd
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def create_frame_landmark_df(results, frame, xyz):
    xyz_skel = xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i,['x','y','z']] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i,['x','y','z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i,['x','y','z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i,['x','y','z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index':'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def do_capture_loop(xyz, duration_per_file=2, overlap=1):
    all_landmarks = []
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = 0
    file_index = 0

    overlap_frames = int(fps * overlap)
    frames_per_file = int(total_frames / (total_frames / (duration_per_file * fps) + overlap_frames))

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        while True:
            frame += 1
            success, image = cap.read()
            if not success:
                print("Failed to read frame.")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            landmarks = create_frame_landmark_df(results, frame, xyz)
            all_landmarks.append(landmarks)

            if frame % frames_per_file == 0 or frame == 1:
                output_file = f'output_{file_index}.parquet'
                combined_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
                combined_landmarks.to_parquet(output_file)
                all_landmarks = []
                file_index += 1
            elif frame % frames_per_file >= frames_per_file - overlap_frames:
                all_landmarks = all_landmarks[1:]

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    if all_landmarks:  # Save any remaining landmarks after the loop ends
        output_file = f'output_{file_index}.parquet'
        combined_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
        combined_landmarks.to_parquet(output_file)
    return all_landmarks

if __name__ == "__main__":
    pq_file = '10042041.parquet'
    xyz = pd.read_parquet(pq_file)
    duration_per_file = 2
    overlap = 1
    landmarks = do_capture_loop(xyz, duration_per_file, overlap)

#test
# import base64
# import dataset
# import tempfile
# import os
#
# def do_capture_loop(xyz, video_base64, duration_per_file=2, overlap=1):
#     all_landmarks = []
#     video_bytes = base64.b64decode(video_base64)
#
#     # Tạo một tệp tạm thời để lưu trữ video
#     temp_file_handle, temp_file_path = tempfile.mkstemp()
#     with open(temp_file_path, "wb") as temp_file:
#         temp_file.write(video_bytes)
#
#     cap = cv2.VideoCapture(temp_file_path)
#     with mp_holistic.Holistic(
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5) as holistic:
#         frame = 0
#         file_index = 0
#         frame_per_file = duration_per_file * cap.get(cv2.CAP_PROP_FPS)
#         overlap_frames = overlap * cap.get(cv2.CAP_PROP_FPS)
#         while cap.isOpened():
#             frame += 1
#             success, image = cap.read()
#             if not success:
#                 print("End of video.")
#                 break
#
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#
#             landmarks = create_frame_landmark_df(results, frame, xyz)
#             all_landmarks.append(landmarks)
#
#             if frame % frame_per_file == 0:
#                 output_file = f'output_{file_index}.parquet'
#                 combined_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
#                 combined_landmarks.to_parquet(output_file)
#                 all_landmarks = []
#                 file_index += 1
#             elif frame % frame_per_file >= frame_per_file - overlap_frames:
#                 all_landmarks = all_landmarks[1:]
#
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#     cap.release()
#
#     # Xóa tệp tạm thời
#     os.close(temp_file_handle)
#     os.unlink(temp_file_path)
#
#     return all_landmarks
#
#
# if __name__ == "__main__":
#     pq_file = '10042041.parquet'
#     xyz = pd.read_parquet(pq_file)
#     video_base64 = dataset.video_base64  # Chuỗi base64 đại diện cho video
#     duration_per_file = 2
#     overlap = 1
#     landmarks = do_capture_loop(xyz, video_base64, duration_per_file, overlap)
#     if landmarks:
#         file_index = 0
#         for i, landmarks_batch in enumerate(landmarks):
#             output_file = f'output_{i}.parquet'
#             landmarks_batch.to_parquet(output_file)
#         file_index += 1

# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy as np
# import time
#
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic
#
# def create_frame_landmark_df(results, frame, xyz):
#     xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
#     face = pd.DataFrame()
#     pose = pd.DataFrame()
#     left_hand = pd.DataFrame()
#     right_hand = pd.DataFrame()
#
#     if results.face_landmarks:
#         for i, point in enumerate(results.face_landmarks.landmark):
#             face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
#     if results.pose_landmarks:
#         for i, point in enumerate(results.pose_landmarks.landmark):
#             pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
#     if results.left_hand_landmarks:
#         for i, point in enumerate(results.left_hand_landmarks.landmark):
#             left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
#     if results.right_hand_landmarks:
#         for i, point in enumerate(results.right_hand_landmarks.landmark):
#             right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
#     face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
#     pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
#     left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
#     right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')
#     landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
#     landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
#     landmarks = landmarks.assign(frame=frame)
#     return landmarks
#
# def do_capture_loop(xyz):
#     all_landmarks = []
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         frame = 0
#         last_capture_time = time.time()
#         while cap.isOpened():
#             frame += 1
#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 continue
#
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#
#             landmarks = create_frame_landmark_df(results, frame, xyz)
#             all_landmarks.append(landmarks)
#
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                                       landmark_drawing_spec=None,
#                                       connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                       landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#
#             cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
#             if cv2.waitKey(5) & 0xFF == 27:
#                 break
#
#             # Check if 2 seconds have passed since last capture
#             if time.time() - last_capture_time >= 1:
#                 last_capture_time = time.time()
#                 pd.concat(all_landmarks).reset_index(drop=True).to_parquet(f'output_{frame}.parquet')
#                 #pd.concat(all_landmarks).reset_index(drop=True).to_parquet(f'output_{int(frame / 30)}.parquet')
#
#     return all_landmarks
#
# if __name__ == "__main__":
#     pq_file = '10042041.parquet'
#     xyz = pd.read_parquet(pq_file)
#     landmarks = do_capture_loop(xyz)

