import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def create_frame_landmark_df(results, frame, xyz):
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates()
    landmarks = []

    for name, landmark_list in zip(['face', 'pose', 'left_hand', 'right_hand'],
                                   [results.face_landmarks, results.pose_landmarks,
                                    results.left_hand_landmarks, results.right_hand_landmarks]):
        if landmark_list:
            for i, point in enumerate(landmark_list.landmark):
                landmarks.append({'x': point.x, 'y': point.y, 'z': point.z,
                                  'type': name, 'landmark_index': i})

    landmarks_df = pd.DataFrame(landmarks)
    landmarks_df = pd.merge(xyz_skel, landmarks_df, on=['type', 'landmark_index'], how='left')
    landmarks_df['frame'] = frame
    return landmarks_df

def do_capture_loop(xyz, video_file_path, duration_per_file=2, overlap=1):
    all_landmarks = []
    cap = cv2.VideoCapture(video_file_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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
                combined_landmarks = pd.concat(all_landmarks, ignore_index=True)
                combined_landmarks.to_parquet(output_file)
                all_landmarks = []
                file_index += 1
            elif frame % frame_per_file >= frame_per_file - overlap_frames:
                all_landmarks = all_landmarks[1:]

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
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
    video_file_path = 'output.mp4'  # Change to your video file path
    duration_per_file = 2
    overlap = 1
    landmarks = do_capture_loop(xyz, video_file_path, duration_per_file, overlap)
    if landmarks:
        for i, landmarks_batch in enumerate(landmarks):
            output_file = f'output_{i}.parquet'
            landmarks_batch.to_parquet(output_file)
