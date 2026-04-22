import cv2

# Załaduj klasyfikator Haar Cascade do detekcji twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Uruchom kamerę (0 = domyślna kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Błąd: Nie można otworzyć kamery.")
    exit()

print("Naciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można odczytać klatki.")
        break

    # Konwersja do skali szarości (wymagana przez Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcja twarzy
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Rysowanie prostokątów wokół wykrytych twarzy
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Wyświetlanie liczby wykrytych twarzy
    cv2.putText(frame, f'Twarze: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Detekcja twarzy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
