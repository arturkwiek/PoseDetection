import cv2
import dlib
import numpy as np

try:
    from mediapipe import solutions as mp_solutions
    MP_AVAILABLE = True
except (ImportError, AttributeError):
    MP_AVAILABLE = False

# Załaduj klasyfikator Haar Cascade do detekcji twarzy
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Załaduj detektor twarzy i model do rozpoznawania 68 punktów charakterystycznych
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks_available = True
except RuntimeError:
    print("Ostrzeżenie: Brak pliku 'shape_predictor_68_face_landmarks.dat'")
    print("Pobierz go z: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    landmarks_available = False

if MP_AVAILABLE:
    mp_pose = mp_solutions.pose
    mp_drawing = mp_solutions.drawing_utils
    pose_tracker = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
else:
    print("Ostrzeżenie: MediaPipe nie jest dostępne, tryby pozy będą wyłączone.")
    print("Zainstaluj: py -m pip install mediapipe")
    pose_tracker = None

# Uruchom kamerę
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    exit()

print("Naciśnij 'q', aby zakończyć.")
print("Naciśnij: 1=TWARZ, 2=LANDMARKS, 3=POSE, 4=ALL")

# Tryby działania
MODE_FACE = "face"
MODE_LANDMARKS = "landmarks"
MODE_POSE = "pose"
MODE_ALL = "all"
mode = MODE_ALL

# Słownik z opisami części twarzy
parts_names = {
    (0, 17): "Kontur twarzy",
    (17, 22): "Lewe brwi",
    (22, 27): "Prawe brwi",
    (27, 36): "Nos",
    (36, 42): "Lewe oko",
    (42, 48): "Prawe oko",
    (48, 61): "Usta",
    (61, 68): "Szczęka"
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można odczytać klatki.")
        break

    # Konwersja do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode in (MODE_FACE, MODE_ALL):
        # ===== TRYB 1: Detekcja twarzy (Haar Cascade) =====
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Rysowanie prostokątów wokół wykrytych twarzy
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Twarz', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Twarze: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    if mode in (MODE_LANDMARKS, MODE_ALL) and landmarks_available:
        # ===== TRYB 2: Detekcja 68 punktów =====
        faces = detector(gray, 1)

        for face in faces:
            # Pobranie 68 punktów charakterystycznych
            landmarks = predictor(gray, face)
            
            # Rysowanie wszystkich 68 punktów
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                # Rysowanie koła w każdym punkcie
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                # Numer punktu (tylko dla oczu)
                if 36 <= i < 48:
                    cv2.putText(frame, str(i), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            # Rysowanie konturów poszczególnych części twarzy
            for (start, end), name in parts_names.items():
                points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                  for i in range(start, end)], dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 0), 2)

        if len(faces) > 0:
            cv2.putText(frame, f'Punkty: {landmarks.num_parts}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if mode in (MODE_POSE, MODE_ALL):
        if pose_tracker is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose_tracker.process(frame_rgb)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 200, 255), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 100, 0), thickness=2
                    ),
                )
                visible_points = sum(
                    1
                    for lm in pose_results.pose_landmarks.landmark
                    if lm.visibility > 0.5
                )
                cv2.putText(
                    frame,
                    f"Pose: {visible_points}/33",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 180, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Pose: brak",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "MediaPipe niedostepne",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    cv2.putText(
        frame,
        f"TRYB: {mode.upper()}  |  1=TWARZ 2=LANDMARKS 3=POSE 4=ALL  Q=KONIEC",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    cv2.imshow('Detekcja twarzy i punktów charakterystycznych', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = MODE_FACE
        print("Tryb: detekcja twarzy")
    elif key == ord('2'):
        mode = MODE_LANDMARKS
        if landmarks_available:
            print("Tryb: 68 punktow twarzy")
        else:
            print("Brak modelu 68 punktow - pozostanie puste okno")
    elif key == ord('3'):
        mode = MODE_POSE
        if pose_tracker is not None:
            print("Tryb: detekcja pozy")
        else:
            print("Brak MediaPipe - tryb pozy niedostepny")
    elif key == ord('4'):
        mode = MODE_ALL
        print("Tryb: ALL")

cap.release()
if pose_tracker is not None:
    pose_tracker.close()
cv2.destroyAllWindows()
