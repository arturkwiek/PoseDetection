import cv2
import dlib
import numpy as np

# Załaduj detektor twarzy i model do rozpoznawania 68 punktów charakterystycznych
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Uruchom kamerę
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    exit()

print("Naciśnij 'q', aby zakończyć.")
print("Detekcja 68 punktów charakterystycznych twarzy...")

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

    # Detekcja twarzy
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
            # Numer punktu
            cv2.putText(frame, str(i), (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Rysowanie konturów poszczególnych części twarzy
        for (start, end), name in parts_names.items():
            points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                              for i in range(start, end)], dtype=np.int32)
            cv2.polylines(frame, [points], False, (255, 0, 0), 2)

        # Wyświetlenie liczby wykrytych twarzy
        cv2.putText(frame, f'Twarze: {len(faces)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Wyświetlenie liczby punktów
        cv2.putText(frame, f'Punkty: {landmarks.num_parts}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Detekcja 68 punktów charakterystycznych twarzy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
