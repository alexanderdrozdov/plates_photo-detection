# plates_photo-detection
Распознавание и чтение автомобильных номеров по фото

Принцип работы:
1) Предобученная yolo11m находит на фото машины, и вырезает их области
2) Модель plate_detection находит на вырезанной области номер и вырезает его
3) Модель best детектирует символы на номере и выводит их на картинку

Модель plate_detection обучена на 11000 фотографий, датасет был взят с roboflow
Модель best обучена на 800 фотографиях

Модели:
https://drive.google.com/drive/folders/1zOlIfDHj4YthH8IboSjwoRbOFQLWpQuM?usp=sharing
