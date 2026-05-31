# receding-hairline
# Receding Hairline Detection - System klasyfikacji wizyjnej

Academic engineering project focused on the automatic classification of the "Receding Hairline" attribute based on facial images from the CelebA dataset. The project demonstrates the evolution of computer vision approaches, comparing classic image processing and descriptor-based machine learning with deep convolutional neural networks (CNN).

## Struktura Projektu

Projekt został zrealizowany sekwencyjnie w ramach pięciu etapów badawczych, z których każdy wnosił usprawnienia do ostatecznego potoku decyzyjnego.

### 1. Przetwarzanie wstępne i filtracja splotowa 
* **Wygładzanie i redukcja szumów:** Implementacja filtrów dolnoprzepustowych (Gaussa oraz medianowego) w celu eliminacji szumów informacyjnych i artefaktów kompresji tekstury skóry.
* **Detekcja krawędzi:** Wykorzystanie filtru Sobela do uwypuklenia pionowych i skośnych krawędzi na styku czoła i linii włosów.
* **Progowanie:** Porównanie progowania globalnego z adaptacyjnym, wykazujące przewagę metod adaptacyjnych w warunkach zmiennego oświetlenia sceny.

### 2. Segmentacja i pomiary geometryczne
* **Próba ekstrakcji ROI:** Konwersja obrazu do przestrzeni barwnej **HSV**, nałożenie maski koloru skóry oraz zastosowanie operacji morfologicznych (otwarcie/zamknięcie) do wyodrębnienia konturu twarzy.
* **Metryki ilościowe:** Wyznaczanie parametrów geometrycznych takich jak pole powierzchni (*Area*), obwód (*Perimeter*) oraz współczynnik kształtu (*Aspect Ratio*).
* **Krytyczna analiza:** Etap ten wykazał niską skuteczność klasycznych metod geometrycznych z powodu "wyciekania" maski na obszary brody, zarostu lub specyficznych fryzur, co uzasadniło przejście do bardziej zaawansowanych deskryptorów.

### 3. Ekstrakcja cech HOG i klasyczna klasyfikacja 
* **Deskryptor teksturalny:** Zastosowanie algorytmu **HOG** (*Histogram of Oriented Gradients*) do opisu rozkładu kierunków gradientów jasności na zbalansowanym zbiorze danych (2000 próbek).
* **Analiza skupisk (PCA):** Redukcja wymiarowości do 2 składowych głównych ujawniła silne nałożenie się klas ("Tak" / "Nie") w przestrzeni cech.
* **Klasyfikatory:** Ewaluacja modeli statystycznych wykazała ograniczoną skuteczność klasycznego podejścia:
  * **SVM (Linear Kernel):** 66.75% dokładności
  * **k-NN (k=5):** 67.75% dokładności

### 4. Głębokie uczenie i architektura CNN
* **Potok danych:** Podział danych metodą stratyfikacji (70% trening, 15% walidacja, 15% test) na zbalansowanym zbiorze 10 000 obrazów.
* **Augmentacja w czasie rzeczywistym:** Zastosowanie `ImageDataGenerator` (losowe obroty, przesunięcia, odbicia lustrzane, modyfikacja jasności) w celu zapobiegania przeuczeniu (*overfitting*).
* **Architektura sieci:** Model sekwencyjny składający się z 3 bloków splotowych (`Conv2D` + `ReLU` + `MaxPooling2D`), warstwy spłaszczającej, warstwy `Dropout (0.5)` oraz wyjściowego neuronu z aktywacją `Sigmoid`.
* **Wyniki testowe:** Model osiągnął stabilną i symetryczną skuteczność na poziomie **84.3% Accuracy** (F1-score: 0.84), znacznie przewyższając metody klasyczne.

---

## 📊 Wyniki Ewaluacji Modelu CNN

### Raport klasyfikacji (Zbiór testowy - 1500 obrazów)

| Klasa | Precision | Recall | F1-score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Brak (0)** | 0.84 | 0.83 | 0.83 | 750 |
| **Zakola (1)** | 0.83 | 0.84 | 0.84 | 750 |
| **Dokładność (Accuracy)** | | | **0.84** | **1500** |

### Analiza błędów i ograniczenia
Weryfikacja wizualna wykazała, że model jest podatny na zjawisko okluzji. W przypadku całkowitego zasłonięcia czoła przez specyficzne fryzury (np. długie grzywki) lub nakrycia głowy (czapki), sieć nie wykrywa krawędzi charakterystycznych dla zakoli, co skutkuje zwróceniem prawdopodobieństwa bliskiego `0.00` (fałszywa pewność klasy "Brak").

---

## Moduł interaktywny (Testowanie własnego zdjęcia)
W projekcie zaimplementowano skrypt pozwalający na załadowanie dowolnego pliku graficznego (JPG/PNG). Obraz wejściowy jest automatycznie normalizowany i skalowany do formatu `128x128x3`, po czym wgrany model dokonuje predykcji i wyświetla wynik wraz z poziomem prawdopodobieństwa w czasie rzeczywistym.


---

## 📦 Wymagane biblioteki
Wszystkie zależności niezbędne do uruchomienia projektu znajdują się w pliku `requirements.txt`.

```bash
pip install -r requirements.txt
