import cv2
import dlib
import numpy as np

# Załaduj klasyfikator Haar Cascade do detekcji twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Załaduj detektor twarzy i model do rozpoznawania 68 punktów charakterystycznych
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks_available = True
except:
    print("Ostrzeżenie: Brak pliku 'shape_predictor_68_face_landmarks.dat'")
    print("Pobierz go z: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    landmarks_available = False

# Uruchom kamerę
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    exit()

print("Naciśnij 'q', aby zakończyć.")
print("Naciśnij 'S', aby przełączyć tryb między detekcją twarzy a 68 punktami.")

# Flaga do przełączania między trybami
mode = "face"  # "face" lub "landmarks"

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

    if mode == "face":
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

        # Wyświetlanie liczby wykrytych twarzy
        cv2.putText(frame, f'Twarze: {len(faces)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, 'TRYB: Detekcja twarzy (Haar Cascade)', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elif mode == "landmarks" and landmarks_available:
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
                # Numer punktu (tylko dla oczy)
                if 36 <= i < 48:  # Punkty oczu
                    cv2.putText(frame, str(i), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            # Rysowanie konturów poszczególnych części twarzy
            for (start, end), name in parts_names.items():
                points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                  for i in range(start, end)], dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 0), 2)

        # Wyświetlenie liczby wykrytych twarzy i punktów
        cv2.putText(frame, f'Twarze: {len(faces)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if len(faces) > 0:
            cv2.putText(frame, f'Punkty: {landmarks.num_parts}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, 'TRYB: 68 punktów charakterystycznych', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Informacja o przełączaniu
    cv2.putText(frame, 'Nacisnij S, aby zmienić tryb', (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Detekcja twarzy i punktów charakterystycznych', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') or key == ord('S'):
        # Przełączanie między trybami
        if mode == "face":
            if landmarks_available:
                mode = "landmarks"
                print("Przełączono na tryb: 68 punktów charakterystycznych")
            else:
                print("Model dla 68 punktów niedostępny!")
        else:
            mode = "face"
            print("Przełączono na tryb: Detekcja twarzy")

cap.release()
cv2.destroyAllWindows()
