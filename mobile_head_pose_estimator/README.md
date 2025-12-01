_Структура mobile_head_pose_estimator:_

Необходимые данные (необходимо добавить):
- imgs - папка с датасетом (раскадрированное видео) (snapshot\_\*.jpg)
- jsons - папка, где лежат snapshot'ы (snapshot\_\*.jspn)
  Важно! json, относящийся к изображению должен быть обозван так же, как и сама картинка (меняется только расширение: jpg -> json)

Папка, которая создается в процессе выполнения кода:
- output - папка с визуализацией (snapshot\_\*\_vis.jpg)

- tests - папка: куда сохраняется картинка, полученная в результате запуска теста, где лежит mock_json с конкретными значениями (нет реального лица)
- requirements.txt
- python файлы


_Основные запускаемые файлы:_

- main1.py
- test_visualization.py


_Основные инструменты:_

- estimator.py
- pose_calculator.py
- visualizer.py
- utils.py


_Для запуска:_

```
pip install -r requirements.txt
```

Для запуска теста:

```
python test_visualization.py
```

Для запуска на реальных данных:
В папке _jsons_ должны лежать json'ы тех данных, на которых требуется запустить код

```
python main1.py
```
