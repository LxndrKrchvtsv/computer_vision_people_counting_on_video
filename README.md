# People Counting & Flow

Computer Vision система подсчёта людей и оценки направления движения на видео
пешеходного перехода. Пайплайн: `image → enhance → segment → clean → detect → decide`.

## Запуск (ВАЖНО)

Используется `.venv` с Python 3.14. Системный `py` (3.13) не видит `cv2`.
**Запускать только интерпретатором из `.venv`:**

```powershell
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe main.py --video data/test.mp4 --display
```

Альтернатива — активировать окружение, затем `python`:

```powershell
.venv\Scripts\Activate.ps1
python main.py --video data/test.mp4 --display
```

Не использовать `py main.py ...` → `ModuleNotFoundError: No module named 'cv2'`.

### Полезные флаги
- `--yolo` — детекция через YOLO вместо контуров.
- `--no-save-stages` — не сохранять покадровые стадии (быстрее).
- `--line-y N` — Y координата линии подсчёта (по умолчанию центр кадра).
- `--max-frames N` — обработать не более N кадров (быстрый прогон, напр. 1 мин ≈ 1800 кадров).
- `--display` — окно предпросмотра.

## Структура
- `src/` — стадии пайплайна (enhance, segment, clean, detect, decision, visualize).
- `main.py`, `src/pipeline.py` — точка входа и сборка пайплайна.
- `data/` — `test.mp4` (вход) и `output/` (результаты).
- `docs/PHASES.md` — план всех фаз и статусы.
- `docs/PHASE_1.md` — детальный план текущей фазы.
