_Основные папки:_

- mydataset - папка с датасетом (раскадрированное видео) (snapshot\_\*.jpg)
- res_my_dataset - папка с json'ами после запуска face_main_ie.snapshot.stage0
- mobile_head_pose_estimator - папка со всеми исполняемыми файлами

_Структура mobile_head_pose_estimator:_

- jsons - папка, где лежат snapshot'ы (snapshot\_\*.jspn)
  Важно! json, относящийся к изображению должен быть обозван так же, как и сама картинка (меняется только расширение: jpg -> json)
- output - папка с визуализацией (snapshot\_\*\_vis.jpg)
- tests - папка: куда сохраняется картинка, полученная в результате запуска теста, где лежит mock_json с конкретными значениями (нет реального лица)
- requirements.txt

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
cd mobile_head_pose_estimator
install -r requirements.txt
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
