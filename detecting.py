import cv2
from ultralytics import YOLO


def main():
    # Загрузка моделей
    car_model = YOLO('yolo11m.pt')
    model_plate = YOLO('plate_detection.pt')
    model_symbols = YOLO('best.pt')

    # Загрузка изображения
    #image_path = "C:/Users/alexanderdrozdov/Pictures/Screenshots/Снимок экрана (6).png"
    image_path = "C:/Users/alexanderdrozdov/Desktop/ввв/g.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Ошибка загрузки изображения")
        return

    cars = car_model(image)
    car_boxes = cars[0].boxes.xyxy.cpu().numpy()

    if len(car_boxes) == 0:
        print("Автомобили не найдены")
        return

    x1_car, y1_car, x2_car, y2_car = map(int, car_boxes[0])

    cv2.rectangle(image, (x1_car, y1_car), (x2_car, y2_car), (255, 0, 0), 2)

    cv2.putText(image, 'car', (x1_car, y2_car - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Детекция номерного знака
    car_roi = image[y1_car:y2_car, x1_car:x2_car]

    plates = model_plate(car_roi)

    if len(plates[0].boxes) == 0:
        print("Номерной знак не найден")
        return

    # Получение координат номера
    x1p, y1p, x2p, y2p = map(int, plates[0].boxes.xyxy[0])

    # Конвертируем в абсолютные координаты
    x1_abs = x1_car + x1p
    y1_abs = y1_car + y1p
    x2_abs = x1_car + x2p
    y2_abs = y1_car + y2p

    cv2.rectangle(image, (x1_abs, y1_abs), (x2_abs, y2_abs), (0, 255, 0), 2)

    cv2.putText(image, 'license plate', (x1_abs, y1_abs-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plate_roi = image[y1_abs:y2_abs, x1_abs:x2_abs]
    characters = model_symbols(plate_roi)
    detected_chars = []

    for box in characters[0].boxes:
        x1c, y1c, x2c, y2c = map(int, box.xyxy[0])
        char = model_symbols.names[int(box.cls)]
        confidence = box.conf.item()

        # Рисуем символы на основном изображении
        abs_x1 = x1_abs + x1c
        abs_y1 = y1_abs + y1c

        #cv2.putText(image, f"{char}", (abs_x1, abs_y1 - 5),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        detected_chars.append((x1c, char))

        # Сборка номера
    detected_chars.sort(key=lambda x: x[0])
    final_number = ''.join([char[1] for char in detected_chars])

    print(f"Распознанный номер: {final_number}")
    cv2.putText(image, final_number, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 4. Сохранение результата
    save_path = "C:/Users/alexanderdrozdov/Desktop/ввв/proccesed_image5.jpg"
    cv2.imwrite(save_path, image)
    print(f"Изображение сохранено: {save_path}")

    # Показать результат (опционально)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    #Отображение машины с выделенным номером
    image_with_plate = image.copy()
    cv2.rectangle(image_with_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('1. Car with License Plate', image_with_plate)
    cv2.waitKey(0)

    # Вырезаем номер
    plate_roi = image[y1:y2, x1:x2]

    # 2. Отображение обрезанного номера
    cv2.imshow('2. Cropped License Plate', plate_roi)
    cv2.waitKey(0)

    # Распознавание символов
    characters = model_symbols(plate_roi)

    # Рисуем распознанные символы на номере
    plate_with_chars = plate_roi.copy()
    detected_chars = []

    for box in characters[0].boxes:
        x1c, y1c, x2c, y2c = map(int, box.xyxy[0])
        char = model_symbols.names[int(box.cls)]
        confidence = box.conf.item()

        # Рисуем bounding box и текст
        cv2.rectangle(plate_with_chars, (x1c, y1c), (x2c, y2c), (0, 0, 255), 1)
        cv2.putText(plate_with_chars, f"{char} ({confidence:.2f})",
                    (x1c, y1c - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
        detected_chars.append((x1c, char))

    # 3. Отображение номера с распознанными символами
    cv2.imshow('3. Character Recognition', plate_with_chars)
    cv2.waitKey(0)

    # Собираем итоговый номер
    detected_chars.sort(key=lambda x: x[0])
    final_number = ''.join([char[1] for char in detected_chars])
    print(f"Распознанный номер: {final_number}")

    # Закрытие всех окон
    cv2.destroyAllWindows()
'''

if __name__ == "__main__":
    main()