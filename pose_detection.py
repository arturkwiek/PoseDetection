import cv2
import numpy as np

try:
    from mediapipe import solutions as mp_solutions
    from mediapipe.framework.formats import landmark_pb2
    import mediapipe as mp
    mp_pose = mp_solutions.pose
    mp_drawing = mp_solutions.drawing_utils
    mp_drawing_styles = mp_solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Błąd MediaPipe: {e}")
    print("Spróbuję użyć alternatywnej metody...")
    MEDIAPIPE_AVAILABLE = False

# Uruchom kamerę
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    exit()

print("Naciśnij 'q', aby zakończyć.")
print("Wykrywanie pozy człowieka...")

if not MEDIAPIPE_AVAILABLE:
    print("Błąd: MediaPipe nie jest dostępny. Instaluję wymaganą wersję...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "mediapipe"])
    try:
        from mediapipe import solutions as mp_solutions
        from mediapipe.framework.formats import landmark_pb2
        import mediapipe as mp
        mp_pose = mp_solutions.pose
        mp_drawing = mp_solutions.drawing_utils
        mp_drawing_styles = mp_solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
    except Exception as e:
        print(f"Nie mogę zainstalować MediaPipe: {e}")
        exit()

# Konfiguracja detektora pozy
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=light, 1=full, 2=heavy
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    # Słownik z nazwami kluczowych punktów pozy
    pose_labels = {
        0: "Nos", 1: "L Oko (wewnętrzne)", 2: "L Oko", 3: "L Oko (zewnętrzne)",
        4: "P Oko (wewnętrzne)", 5: "P Oko", 6: "P Oko (zewnętrzne)",
        7: "L Ucho", 8: "P Ucho",
        9: "L Ramię", 10: "P Ramię",
        11: "L Łokieć", 12: "P Łokieć",
        13: "L Nadgarstek", 14: "P Nadgarstek",
        15: "L Mały Palec", 16: "P Mały Palec",
        17: "L Indeks", 18: "P Indeks",
        19: "L Środek", 20: "P Środek",
        23: "L Biodro", 24: "P Biodro",
        25: "L Kolano", 26: "P Kolano",
        27: "L Kostka", 28: "P Kostka",
        29: "L Pieta", 30: "P Pieta",
        31: "L Palec u nogi", 32: "P Palec u nogi"
    }

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie można odczytać klatki.")
            break

        frame_count += 1
        height, width, _ = frame.shape

        # Konwersja BGR do RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = pose.process(frame_rgb)

        # Jeśli zostały wykryte punkty pozy
        if results.pose_landmarks:
            # Rysowanie szkieletu (skeleton)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Dodatkowe informacje: rysowanie punktów z ich numerami i współrzędnymi
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility > 0.5:  # Jeśli punkt jest widoczny
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    
                    # Rysowanie koła w punkcie
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    # Wyświetlenie numeru punktu
                    cv2.putText(frame, str(idx), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Informacje na ekranie
        cv2.putText(frame, 'Detekcja pozy - MediaPipe Pose', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results.pose_landmarks:
            # Liczenie widocznych punktów
            visible_points = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
            cv2.putText(frame, f'Widoczne punkty: {visible_points}/33', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Współrzędne głowy (Nos - punkt 0)
            nose = results.pose_landmarks.landmark[0]
            if nose.visibility > 0.5:
                x = int(nose.x * width)
                y = int(nose.y * height)
                cv2.putText(frame, f'Glowa: ({x}, {y})', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Brak detekcji pozy', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # FPS
        cv2.putText(frame, f'Klatka: {frame_count}', (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Detekcja pozy', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Program zakończony.")
