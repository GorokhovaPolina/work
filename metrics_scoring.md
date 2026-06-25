# Метрики оценивания качества прохождения инструкций

## Интегральная оценка одной инструкции (`InstructionScore`)

### Диапазон и семантика

`total_score ∈ [-1.0, 1.0]`, начало в 0:

| Диапазон | Смысл |
|---|---|
| `(0.85, 1.0]` | Уверенное выполнение (Accept — зелёный) |
| `(0.75, 0.85]` | Хорошее выполнение (Accept — светло-зелёный) |
| `(0.65, 0.75]` | Выше среднего (Wait — светло-зелёный) |
| `(0.50, 0.65]` | Среднее (Wait — жёлтый) |
| `[0.00, 0.50]` | Низкое (Wait/Decline — красный) |
| `(-1.0, 0.00)` | Нарушение: неверное направление / круговое движение / пауза |

### Формула

```
total_score = clamp(positive_score − negative_score, −1, 1)

positive_score = w_progress   · main_progress
               + w_compliance · compliance

negative_score = w_ortho      · ortho_velocity_penalty
               + w_ortho_abs  · ortho_abs_penalty
               + w_stall      · stall_penalty
               + w_wrong      · wrong_penalty
               + w_trajectory · trajectory_penalty
```

Максимум `positive_score = w_progress + w_compliance = 0.75 + 0.25 = 1.0`  
Минимум `negative_score = w_ortho + w_ortho_abs + w_stall + w_wrong + w_trajectory = 0.30 + 0.25 + 0.30 + 0.30 + 0.70 = 1.85`  
→ минимально возможный `total_score ≈ −1.0` (clamp)

### Компоненты

**Все компоненты ∈ [0, 1].**

#### `main_progress` — прогресс по главной оси

```
main_progress = clamp(max_deviation / target_angle, 0, 1)
```

`target_angle`: 25° для Left/Right, 15° для Up, 12° для Down.  
`max_deviation` — максимальное угловое отклонение от стартовой позы в правильном направлении, накопленное за всё время инструкции.

`reached_target = max_deviation ≥ target_angle × 0.75`

#### `compliance` — монотонность движения

```
compliance = correct_frames / (total_frames − 1)
```

Кадр считается правильным, если угловая скорость по главной оси ≥ `direction_noise = 1.5°/с`.  
Скорость вычисляется поштучно: `v_main = delta_main / dt_sec`, где `dt_sec` — время между кадрами в секундах. Это обеспечивает **независимость от fps** (работает от 2 до 60 fps).

#### `ortho_velocity_penalty` — квадратичный штраф за скорость по ортогональной оси

```
ortho_velocity_penalty = clamp(integral(ratio²·dt) / T, 0, 1)
ratio = clamp(v_ortho / ortho_norm, 0, 1),  ortho_norm = 5°/с
```

Квадратичная зависимость: малые дрожания (v_ortho ≈ 1°/с) дают ratio² ≈ 0.04, большие движения (v_ortho = 5°/с) дают ratio² = 1. Интеграл нормируется на суммарное время T → результат в [0, 1].

#### `ortho_abs_penalty` — штраф за постоянный наклон по ортогональной оси

```
ortho_abs_penalty = clamp(max_ortho_abs / ortho_abs_norm, 0, 1)
ortho_abs_norm = 10°
```

Ловит случай, когда голова зафиксирована с большим углом по ортогональной оси (например, pitch = 16° при повороте влево/вправо), но `v_ortho = 0`. Скоростной штраф такое не замечает; этот — замечает.

#### `stall_penalty` — штраф за паузу

```
stall_penalty = clamp((stall_sec − T_start) / (T_max − T_start), 0, 1)
T_start = 3.0с,  T_max = 6.0с
```

`stall_sec` — длина максимальной непрерывной паузы (периода, когда `v_main < direction_noise`).  
Пауза сбрасывается, как только движение возобновляется.

**Почему 3.0с:** нормальное выполнение инструкции занимает 0.5–1.5с. Реальный человек на любом устройстве (2–60 fps) не будет стоять дольше 2с. Статичное фото/видео зависает навсегда — штраф нарастает от 3с до максимума к 6с.

#### `wrong_penalty` — штраф за движение в неверную сторону

```
wrong_penalty = clamp(sum_main_negative / wrong_norm, 0, 1)
wrong_norm = 15°
```

`sum_main_negative` — суммарный угол (интеграл по времени), пройденный в направлении, противоположном инструкции.

#### `trajectory_penalty` — штраф за круговое/лемнискатное движение

```
trajectory_penalty = clamp(sum_ortho_path / (max(max_deviation, 1°) · ratio_norm), 0, 1)
ratio_norm = 0.80
```

`sum_ortho_path` — суммарный путь по ортогональной оси (сумма `|delta_ortho|` по всем кадрам).

Физический смысл: при прямолинейном повороте голова движется только по главной оси — `sum_ortho_path ≈ 0`, штраф ≈ 0. При круговом движении (pitch и yaw растут вместе) ортогональный путь сопоставим с `max_deviation` — штраф → 1. При лемнискате pitch совершает полный оборот — суммарный ортогональный путь велик даже при нулевом конечном отклонении — штраф → 1.

Знаменатель `max_deviation` (а не `sum_main_positive`) выбран намеренно: это реальный масштаб достигнутого отклонения, не зависящий от fps.

### Веса

| Компонент | Вес | Тип |
|---|---|---|
| `w_progress` | 0.75 | позитивный |
| `w_compliance` | 0.25 | позитивный |
| `w_ortho` | 0.30 | штраф (скорость) |
| `w_ortho_abs` | 0.25 | штраф (абсолютный угол) |
| `w_stall` | 0.30 | штраф (пауза) |
| `w_wrong` | 0.30 | штраф (неверное направление) |
| `w_trajectory` | 0.70 | штраф (круговое движение) |

### Инструкция Straight

Для Straight формула упрощена:

```
compliance = correct_frames / (total_frames − 1)
total_score = clamp(2 · compliance − 1, −1, 1)
```

Кадр правильный: `|yaw| + |pitch| ≤ 7°` (нейтральная зона). Счёт идёт по одному условию — двойного инкремента нет.

---

## Агрегация по сессии

```
session_pass_ratio   = passed_instructions / total_instructions
session_mean_score   = mean(all instruction scores)
session_ce_score     = mean(top 80% instruction scores)

overall_session_score = 0.45 · session_mean_score
                      + 0.35 · session_ce_score
                      + 0.20 · session_pass_ratio
```

| `overall_session_score` | `session_pass_ratio` | Вердикт |
|---|---|---|
| ≥ 0.85 | ≥ 0.90 | Отлично |
| ≥ 0.75 | ≥ 0.80 | Хорошо (рекомендуемый уровень) |
| ≥ 0.65 | ≥ 0.70 | Слабо (требует внимания) |
| < 0.65 | < 0.70 | Не принято |

---

## Ранний выход

**Accept** (досрочно): 30 кадров подряд `total_score ≥ 0.85`.  
**Decline** (досрочно): 60 кадров подряд `total_score` в красной зоне (`< 0.50`).

---

## Цвет маски по скору

| `total_score` | R | G | B | Тип |
|---|---|---|---|---|
| `(0.85, 1.0]` | 0 | 255 | 0 | Уверенный |
| `(0.75, 0.85]` | 85 | 255 | 85 | Хороший |
| `(0.65, 0.75]` | 170 | 255 | 170 | Выше среднего |
| `(0.50, 0.65]` | 255 | 255 | 0 | Средний |
| `[0.00, 0.50]` | 255 | 0 | 0 | Низкий |
| `< 0` | 255 | 0 | 0 | Нарушение |

`stable_time` растёт при `score > 0.65`, не изменяется при `(0.50, 0.65]`, обнуляется при `score ≤ 0.50`.

---

## Метрики на выходе ядра (per frame)

1. `instruction_type` — тип текущей инструкции
2. `total_score` (float, `[-1, 1]`) — текущий скор
3. `main_progress` (float, `[0, 1]`) — прогресс по главному углу
4. `compliance` (float, `[0, 1]`) — доля правильных кадров
5. `ortho_penalty` (float, `[0, 1]`) — скоростной штраф ортогональной оси
6. `stall_penalty` (float, `[0, 1]`) — штраф паузы
7. `wrong_penalty` (float, `[0, 1]`) — штраф неверного направления
8. `reached_target` (bool) — достигнуто ли минимальное отклонение
9. `InstructionProcessVerdict` — Accept / Decline / Wait
