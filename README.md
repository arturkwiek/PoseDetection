# PoseDetection — detekcja twarzy, punktów charakterystycznych i pozy

Zestaw niezależnych skryptów do analizy sylwetki i twarzy na obrazie. Każdy plik da się
uruchomić osobno — to raczej **poligon technik** niż jeden spójny program.

## Skrypty

| Plik | Co robi |
|---|---|
| `face_detection.py` | Wykrywanie twarzy klasyfikatorem **Haara** (`haarcascade_frontalface_default.xml`) — metoda klasyczna, szybka, bez sieci neuronowej |
| `facial_landmarks_68_points.py` | **68 punktów charakterystycznych** twarzy z modelu dlib (`shape_predictor_68_face_landmarks.dat`) — kontur, brwi, oczy, nos, usta |
| `pose_detection.py` | Detekcja pozy sylwetki |
| `face_detection_combined.py` | Połączenie powyższych podejść |

Dane i wyniki: `20260507_063354.jpg` (zdjęcie testowe), katalog `wynik/`.

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python dlib numpy
python face_detection.py
```

Uwaga: `dlib` bywa kłopotliwy w instalacji na Windows — wymaga kompilatora C++.
Jeśli nie przechodzi, prostszym wyjściem jest `mediapipe` zamiast dlib.

Oba pliki modeli (`.xml` i `.dat`) leżą w repozytorium, więc skrypty działają bez pobierania
czegokolwiek z sieci.

## Powiązania

Temat pokrywa się z dwoma dojrzalszymi projektami:

- **`Facial-Emotion-Recognition/EmoFACS`** — od punktów charakterystycznych idzie dalej,
  do Action Units i emocji,
- **`NCBiR/AgentPos`** — detekcja pozy wielu osób przez YOLO Pose, z porównaniem
  sześciu konfiguracji.

Ten katalog traktować jako miejsce, gdzie te techniki były sprawdzane pojedynczo.

## Status

🟡 **Wstrzymany.** Ostatni commit: kwiecień 2026, trzy pliki niezacommitowane od maja.
