import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def create_frame_landmark_df(results,frame,xyz):
  xyz_skel = xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True) \
            .copy()
  face = pd.DataFrame()
  pose = pd.DataFrame()
  left_hand = pd.DataFrame()
  right_hand = pd.DataFrame()

  if results.face_landmarks:
    for i,point in enumerate(results.face_landmarks.landmark):
      face.loc[i,['x','y','z']] = [point.x, point.y,point.z]
  if results.pose_landmarks:
    for i,point in enumerate(results.pose_landmarks.landmark):
      pose.loc[i,['x','y','z']] = [point.x, point.y,point.z]
  if results.left_hand_landmarks:
    for i,point in enumerate(results.left_hand_landmarks.landmark):
      left_hand.loc[i,['x','y','z']] = [point.x, point.y,point.z]
  if results.right_hand_landmarks:
    for i,point in enumerate(results.right_hand_landmarks.landmark):
      right_hand.loc[i,['x','y','z']] = [point.x, point.y, point.z]
  face = face.reset_index()\
      .rename(columns={'index':'landmark_index'})\
      .assign(type='face')
  pose = pose.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='pose')
  left_hand = left_hand.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='left_hand')
  right_hand = right_hand.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='right_hand')
  landmarks = pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True)
  landmarks = xyz_skel.merge(landmarks,on=['type','landmark_index'], how='left')
  landmarks = landmarks.assign(frame=frame)
  return landmarks
def do_capture_loop(xyz, videopath,duration_per_file=2, overlap=1):
    all_landmarks = []
    cap = cv2.VideoCapture(videopath)  # Mở camera laptop
    with mp_holistic.Holistic(
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5) as holistic:
        frame = 0
        file_index = 0
        frame_per_file = duration_per_file * cap.get(cv2.CAP_PROP_FPS)
        overlap_frames = overlap * cap.get(cv2.CAP_PROP_FPS)
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

            if frame % frame_per_file == 0:
                output_file = f'output_{file_index}.parquet'
                combined_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
                combined_landmarks.to_parquet(output_file)
                all_landmarks = []
                file_index += 1
            elif frame % frame_per_file >= frame_per_file - overlap_frames:
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
    return all_landmarks

if __name__ == "__main__":
    pq_file = '10042041.parquet'
    xyz = pd.read_parquet(pq_file)
    duration_per_file = 2
    overlap = 1
    videopath = '442489467_7479661802141368_6060868340835589970_n.mp4'
    landmarks = do_capture_loop(xyz,videopath, duration_per_file, overlap)
    if landmarks:
        file_index = 0
        for i, landmarks_batch in enumerate(landmarks):
            output_file = f'output_{i}.parquet'
            landmarks_batch.to_parquet(output_file)
        file_index += 1
