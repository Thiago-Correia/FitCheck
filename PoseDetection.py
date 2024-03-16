import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
shoulderAngleConnections_style = mp_drawing_styles.get_default_pose_landmarks_style()
armAngleConnections_style = mp_drawing_styles.get_default_pose_landmarks_style()
backAngleConnections_style = mp_drawing_styles.get_default_pose_landmarks_style()
legAngleConnections_style = mp_drawing_styles.get_default_pose_landmarks_style()


custom_connections = list(mp_pose.POSE_CONNECTIONS)
shoulderAngleConnections = list(mp_pose.POSE_CONNECTIONS)
armAngleConnections = list(mp_pose.POSE_CONNECTIONS)
backAngleConnections = list(mp_pose.POSE_CONNECTIONS)
legAngleConnections = list(mp_pose.POSE_CONNECTIONS)

rosca_Landmarks = [
    #Face
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    
    #RightArm
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_PINKY,
    
    #RightLeg
    PoseLandmark.RIGHT_FOOT_INDEX,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_HIP,
]

for landmark in rosca_Landmarks:
    custom_style[landmark] = mp_drawing.DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
    custom_connections = [connection_tuple for connection_tuple in custom_connections if landmark.value not in connection_tuple]

shoulderAngle_Landmarks = [
    #Face
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    
    #RightArm
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_PINKY,
    
    #LeftArm
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.LEFT_PINKY,
    
    #RightLeg
    PoseLandmark.RIGHT_FOOT_INDEX,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_HIP,
    
    #LeftLeg
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.LEFT_HEEL,
]

for landmark in shoulderAngle_Landmarks:
    shoulderAngleConnections_style[landmark] = mp_drawing.DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
    shoulderAngleConnections = [connection_tuple for connection_tuple in shoulderAngleConnections if landmark.value not in connection_tuple]


armAngle_Landmarks = [
    #Face
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    
    #RightArm
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_PINKY,
    
    #LeftArm
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.LEFT_PINKY,
    
    #RightLeg
    PoseLandmark.RIGHT_FOOT_INDEX,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_HIP,
    
    #LeftLeg
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_HIP, 
]

for landmark in armAngle_Landmarks:
    armAngleConnections_style[landmark] = mp_drawing.DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
    armAngleConnections = [connection_tuple for connection_tuple in armAngleConnections if landmark.value not in connection_tuple]

backAngle_Landmarks = [
    #Face
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    
    #RightArm
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_PINKY,
    
    #LeftArm
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.LEFT_PINKY,
    
    #RightLeg
    PoseLandmark.RIGHT_FOOT_INDEX,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_HIP,
    
    #LeftLeg
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.LEFT_HEEL,
]

for landmark in backAngle_Landmarks:
    backAngleConnections_style[landmark] = mp_drawing.DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
    backAngleConnections = [connection_tuple for connection_tuple in backAngleConnections if landmark.value not in connection_tuple]

legAngle_Landmarks = [
    #Face
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    
    #RightArm
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_PINKY,
    
    #LeftArm
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.LEFT_PINKY,
    
    #RightLeg
    PoseLandmark.RIGHT_FOOT_INDEX,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_HIP,
    
    #LeftLeg
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.LEFT_HEEL,
]

for landmark in legAngle_Landmarks:
    legAngleConnections_style[landmark] = mp_drawing.DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
    legAngleConnections = [connection_tuple for connection_tuple in legAngleConnections if landmark.value not in connection_tuple]





# VIDEO FEED
cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

armAngle = 0
shoulderAngle = 0
backAngle = 0
legAngle = 0



## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angle
            armAngle = calculate_angle(shoulder, elbow, wrist)
            shoulderAngle = calculate_angle(shoulder, elbow, hip)
            backAngle = calculate_angle(shoulder, hip, knee)
            legAngle = calculate_angle(hip, knee, ankle)
            
            
            # Visualize angle
            # cv2.putText(image, str(legAngle), 
            #                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )

                
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, connections = custom_connections,
                                landmark_drawing_spec = custom_style, 
                                connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2) 
                                 )
        
        # Curl counter logic
        if shoulderAngle > 180 or shoulderAngle < 170:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, shoulderAngleConnections,
                                landmark_drawing_spec = shoulderAngleConnections_style, 
                                connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                )
        if armAngle > 180 or armAngle < 0:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, armAngleConnections,
                                landmark_drawing_spec = armAngleConnections_style, 
                                connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                )
        if backAngle < 173:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, backAngleConnections,
                                landmark_drawing_spec = backAngleConnections_style, 
                                connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                )
        if legAngle > 176 or legAngle < 153:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, legAngleConnections,
                                landmark_drawing_spec = legAngleConnections_style, 
                                connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                )
        # if counter == 2:
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, custom_connections,
        #                         mp_drawing.DrawingSpec(color=(245,117,250), thickness=2, circle_radius=2), 
        #                         mp_drawing.DrawingSpec(color=(245,66,0), thickness=2, circle_radius=2) 
        #                          )
        
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




