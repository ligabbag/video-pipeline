# Video pipeline: JSON-сценарий → длинный ролик 16:9 + Shorts 9:16 → YouTube

Конвейер берёт сценарий (контракт `video_script.schema.json`), озвучивает его через ElevenLabs,
генерирует картинки и обложку, рендерит ролик на FFmpeg, нарезает вертикальные Shorts и загружает всё на YouTube.
Сценарий конвейер **никогда не изменяет**: всё, что он узнаёт в процессе, пишется в `state/<video_id>.json`.

```
python -m pipeline validate <script.json>
python -m pipeline run <script.json> [--stages tts,images,thumbnail,render,shorts] [--mock] [--force]
python -m pipeline upload <script.json> [--privacy private|unlisted|public] [--publish-at ISO8601]
                                        [--no-shorts] [--dry-run] [--force-upload]
python -m pipeline status <video_id>
```

## 1. Установка

Нужны Python 3.12 и FFmpeg (с libx264, libmp3lame и libass — в сборках brew и apt они есть).

**macOS**

```bash
brew install ffmpeg python@3.12
```

**Ubuntu**

```bash
sudo apt update
sudo apt install -y ffmpeg python3.12 python3.12-venv fonts-dejavu-core
```

**Виртуальное окружение и зависимости** (из папки проекта):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Проверка, что всё на месте:

```bash
ffmpeg -version
python -m pytest          # ~5 минут: unit-тесты + mock-прогоны; интернет и ключи не нужны
```

## 2. Первый запуск без ключей (mock)

```bash
python -m pipeline validate video_script.example.json
python -m pipeline run video_script.example.json --mock
python -m pipeline status tech-0001-knight-capital
```

`--mock` не ходит ни в один API: вместо голоса — тишина нужной длины с фейковыми таймингами слов
(по `pacing.wpm`), вместо картинок — цветные заглушки с номером сцены, вместо музыки — тихий синтезированный
гул (если в `assets/` пусто). Зато весь остальной конвейер настоящий: таймлайн, рендер, переходы, субтитры,
стемы, обложка, Shorts. Результат — в `output/tech-0001-knight-capital/`.

`--mock` разрешает любой `status` сценария. Без `--mock` рендерятся только сценарии со `status: "approved"`.

## 3. Реальный запуск

1. **Ключи.** Скопируйте `.env.example` в `.env` и впишите ключи:

   ```bash
   cp .env.example .env
   ```

   - `ELEVENLABS_API_KEY` — обязателен;
   - `BFL_API_KEY`, если в сценарии `image_provider: "flux"`, или `OPENAI_API_KEY`, если `"gpt_image"`.
     Провайдера можно переопределить: `--image-provider gpt_image` или `PIPELINE_IMAGE_PROVIDER` в `.env`.

   `.env` уже в `.gitignore`. Ключи читаются только из окружения или `.env` и не попадают в логи и state.

2. **Музыка.** Скачайте треки из [YouTube Audio Library](https://studio.youtube.com/) (YouTube Studio →
   «Фонотека») и разложите по настроениям — по папке на каждый `music_mood` из сценария:

   ```
   assets/music/tense_minimal/      assets/music/curious_pulse/     assets/music/dread_drone/
   assets/music/uplifting_resolve/  assets/music/reflective_piano/
   ```

   Форматы: mp3, wav, m4a, flac, ogg. Достаточно одного трека на папку: он зациклится на длину секции.
   Если треков несколько, выбор детерминирован (один и тот же для данного ролика и настроения).
   Нет трека — этот участок останется без музыки, в логе будет предупреждение.

3. **Звуковые эффекты.** `assets/sfx/<name>.wav`, имена из сценария:
   `low_boom`, `whoosh`, `error_beep`, `alarm_rise`, `typing`, `cash_register`.

4. **Шрифт субтитров.** Положите TTF шрифта из `subtitles.font` (в примере — Montserrat ExtraBold,
   бесплатный на Google Fonts) в `assets/fonts/`. Без него libass подставит системный шрифт.

5. **Сценарий** должен пройти валидатор и иметь `status: "approved"` (утверждение — действие человека):

   ```bash
   python -m pipeline validate scripts/my_video.json
   python -m pipeline run scripts/my_video.json
   ```

   На macOS можно ускорить кодирование: `--hw` (h264_videotoolbox).

Удобно идти по этапам и смотреть промежуточный результат (и расходы):

```bash
python -m pipeline run scripts/my_video.json --stages tts        # озвучка + таймлайн
python -m pipeline run scripts/my_video.json --stages images     # картинки и диаграммы
python -m pipeline run scripts/my_video.json --stages thumbnail  # обложка final/thumbnail.jpg
python -m pipeline run scripts/my_video.json --stages render     # длинный ролик, стемы, SRT, metadata.json
python -m pipeline run scripts/my_video.json --stages shorts     # Shorts
python -m pipeline status <video_id>
```

## 4. Что получится

```
output/<video_id>/
  audio/s001.mp3 …          озвучка по сценам (+ hook_<short_id>.mp3)
  images/s001.png …         картинки сцен (≥ 1,25× кадра под Ken Burns) и диаграммы 1920×1080
  images/thumbnail_bg.png   картинка для обложки (без текста)
  clips/                    клипы сцен (clips/shorts/<short_id>/ — для Shorts)
  stems/voice.wav music.wav sfx.wav mix.wav     стемы 48 кГц и итоговый микс (−14 LUFS, −1 dBTP)
  final/long.mp4            длинный ролик: H.264 CRF 20, AAC 192k, faststart
  final/thumbnail.jpg       обложка 1280×720, JPEG меньше 2 МБ, с текстом из thumbnail.text_overlay
  final/shorts/<short_id>.mp4                   1080×1920 с вшитыми субтитрами
  captions/en.srt           субтитры: ≤ 42 символов в строке, ≤ 2 строк
  metadata.json             title, description (главы и источники подставлены), теги, флаги
  pipeline.log              полный лог
  work/                     временные файлы (можно удалять)
state/<video_id>.json       состояние: статусы этапов, тайминги сцен и слов, хеши, расходы, что загружено на YouTube
cache/                      кеш ответов API (аудио и картинки) — не удаляйте, это деньги
secrets/                    client_secret.json и токены каналов (в git и Docker-образ не попадает)
```

**Обложка.** Картинка генерируется тем же провайдером, что и сцены (`thumbnail.image_prompt` + общий стиль),
а текст `thumbnail.text_overlay` рисует сам конвейер — шрифтом из `assets/fonts`, цветом `text_overlay_color`,
с обводкой и тёмной подложкой. Буквы — не меньше 12 % высоты кадра; текст не выходит за поля и не заходит в
правый нижний угол, где YouTube показывает длительность. Если текст слишком длинный для таких букв, в логе
будет предупреждение — сократите его до 3–5 слов. Поменяли только текст — картинка заново не оплачивается.

## 5. Загрузка на YouTube

```bash
python -m pipeline upload scripts/my_video.json --dry-run     # показать, что будет загружено (без сети)
python -m pipeline upload scripts/my_video.json               # загрузить всё как private
```

Команда загружает длинный ролик, ставит обложку, добавляет английские субтитры и затем загружает каждый Short
со ссылкой на длинный ролик в описании. Она отказывается работать (код выхода 2), пока сценарий не утверждён
(`status: "approved"`, `qa.facts_verified: true`) и пока `python -m pipeline run` не завершил все этапы.

Один раз нужно дать конвейеру доступ к каналу. Это бесплатно и занимает минут пятнадцать. Нужен аккаунт Google,
которому принадлежит YouTube-канал. Названия кнопок ниже — из английского интерфейса консоли (язык меняется
в ⋮ → Preferences); Google время от времени переставляет пункты, так что ориентируйтесь и на описания.

### Шаг 1. Создать проект в Google Cloud Console

1. Откройте <https://console.cloud.google.com/> и войдите.
2. *Что на экране:* вверху синяя полоса; слева от строки поиска — кнопка выбора проекта (на ней написано
   «Select a project» или имя текущего проекта). Нажмите её.
3. В открывшемся окне справа вверху — **New project**. Имя любое, например `video-pipeline`; поле
   Location (Organization) оставьте как есть — «No organization». Нажмите **Create**.
4. Через несколько секунд в колокольчике (справа вверху) появится уведомление — нажмите в нём
   **Select project**. Проверьте, что на кнопке выбора проекта теперь стоит `video-pipeline`.

### Шаг 2. Включить YouTube Data API v3

1. Меню ☰ (слева вверху) → **APIs & Services** → **Library**.
2. *Что на экране:* витрина с карточками и большой строкой поиска. Наберите `YouTube Data API v3`, откройте
   карточку с красным значком YouTube.
3. Нажмите синюю кнопку **Enable**. Когда на её месте появится **Manage**, API включён.

Не перепутайте с «YouTube Analytics API» и «YouTube Reporting API» — они не нужны.

### Шаг 3. Экран согласия OAuth: тип External и вы в test users

1. Меню ☰ → **APIs & Services** → **OAuth consent screen**. Откроется раздел **Google Auth Platform**.
2. *Первый раз:* посередине кнопка **Get started**. Мастер из четырёх шагов:
   - **App Information** — App name (например, `video-pipeline`) и ваш адрес в User support email → Next;
   - **Audience** — выберите **External** → Next (Internal доступен только организациям Google Workspace);
   - **Contact Information** — ваш адрес → Next;
   - **Finish** — галочка согласия с Google API Services: User Data Policy → **Create**.
3. В меню слева откройте **Audience**. *Что на экране:* блок «Publishing status» со словом **Testing** и блок
   «User type: External». Ниже — раздел **Test users**. Нажмите **+ Add users**, впишите адрес Google-аккаунта,
   под которым вы управляете каналом, и сохраните.

Без этого шага Google при входе ответит «Error 403: access_denied» — в режиме Testing пускают только
перечисленных пользователей. Раздел «Data Access» (scopes) заполнять не нужно: конвейер сам запросит одно
разрешение — управление видео и субтитрами на YouTube (`youtube.force-ssl`).

### Шаг 4. Создать OAuth client типа Desktop app и скачать JSON

1. В том же разделе Google Auth Platform откройте слева **Clients** → **+ Create client**.
2. **Application type** → **Desktop app**. Имя любое. Нажмите **Create**.
3. *Что на экране:* окно «OAuth client created» с Client ID. Нажмите **Download JSON** сразу: позже консоль
   может не показать секрет повторно (тогда в карточке клиента добавьте новый секрет и скачайте JSON заново).
4. Положите файл в папку `secrets/` проекта под именем `client_secret.json`:

   ```bash
   mkdir -p secrets
   mv ~/Downloads/client_secret_*.json secrets/client_secret.json
   ```

Нужен именно тип **Desktop app**: для него ничего не надо настраивать в redirect URI. Если скачать клиент типа
«Web application», конвейер скажет об этом и не начнёт вход. Папка `secrets/` уже в `.gitignore` и
`.dockerignore`. Другой путь к файлу задаётся в `.env`: `YOUTUBE_CLIENT_SECRET_FILE`.

### Шаг 5. Сухой прогон

```bash
python -m pipeline upload scripts/my_video.json --dry-run
```

Ничего не отправляет, не открывает браузер и не требует `client_secret.json`. Печатает: какие файлы будут
загружены, заголовок, описание целиком, теги, категорию, язык, флаги «для детей» и «синтетический контент»,
приватность, время публикации, что будет с обложкой и субтитрами, названия и описания Shorts, сколько квоты это
стоит. Прочитайте описание и названия глазами — после загрузки эта команда их уже не меняет.

Если вместо плана напечатано `refused:` и список причин — это предусловия: сценарий не утверждён, какой-то этап
не завершён, файла нет, сценарий меняли после рендера. Рядом с каждой причиной написано, что сделать.

### Шаг 6. Первая загрузка — в private

```bash
python -m pipeline upload scripts/my_video.json
```

1. Откроется браузер (если нет — скопируйте ссылку из терминала в браузер на этом же компьютере).
2. Выберите аккаунт Google. Если у аккаунта несколько каналов, Google спросит, какой использовать, —
   выберите нужный.
3. *Что на экране:* предупреждение «Google hasn't verified this app» («Приложение не проверено Google»).
   В режиме Testing на нём есть кнопка **Continue**. После перевода в In production (шаг 7) кнопки нет —
   нажмите **Advanced** («Дополнительно») и затем **Go to video-pipeline (unsafe)**
   («Перейти на страницу video-pipeline (небезопасно)»). Это ваше собственное приложение, так что всё в порядке.
4. Подтвердите доступ. В браузере появится «Done. You can close this tab», а в терминале — рамка:

   ```
   YouTube channel for 'yt_tech_disasters_en': Tech Disasters  (ID UCxxxxxxxxxxxxxxxxxxxxxx)
   ```

   **Проверьте, что это тот канал.** Название и ID сохраняются в `secrets/token_<channel_id>.json`; при каждом
   следующем запуске конвейер сверяет их и отказывается работать, если токен оказался от другого канала.
   Вошли не туда — удалите этот файл и запустите команду ещё раз.
5. Дальше идёт загрузка с процентами в логе. В конце — ссылки на ролики и, если есть Shorts, напоминание
   (см. «После загрузки» ниже).

По умолчанию всё загружается как **private**: откройте ролик в [YouTube Studio](https://studio.youtube.com/),
убедитесь, что всё на месте, и опубликуйте его там. Когда доверие к процессу появится:

```bash
python -m pipeline upload scripts/my_video.json --privacy unlisted
python -m pipeline upload scripts/my_video.json --privacy public
python -m pipeline upload scripts/my_video.json --publish-at 2026-12-01T18:00:00+03:00
python -m pipeline upload scripts/my_video.json --no-shorts       # только длинный ролик
python -m pipeline status <video_id>                              # ссылки на всё загруженное
```

`--publish-at` — отложенная публикация. Время обязательно с часовым поясом (`+03:00` или `Z`) и в будущем,
иначе команда откажется. Ролик загружается как private (иначе YouTube не применит расписание) и сам станет
публичным в указанный момент; Shorts получают то же время.

**Повторный запуск безопасен.** В state записано, что уже на YouTube: ID роликов, стоит ли обложка, загружены
ли субтитры. Команда догружает только недостающее. Оборвался интернет посреди ролика — запустите её ещё раз,
загрузка продолжится с того же места. Упала только обложка — в следующий раз уйдёт только обложка.
`--force-upload` загружает всё заново **новыми** видео; старые он не удаляет, а предупреждает о дубле —
удалите их в Studio сами. Он же нужен, если ролик удалён в Studio или перерендерен после загрузки: конвейер
помнит прежний ID и сам ничего не подменяет. Вместе с `--no-shorts` перезаливается только длинный ролик.

### Шаг 7. Про 7 дней: переведите приложение в In production

Пока приложение в статусе **Testing**, Google выдаёт refresh-токен, который **перестаёт работать через 7 дней**.
Через неделю команда остановится с сообщением про `invalid_grant`, удалит токен и попросит войти заново. Для
ручных запусков это терпимо, для автономной работы — нет.

Чтобы токен жил долго:

1. Google Cloud Console → Google Auth Platform → **Audience**.
2. В блоке «Publishing status» нажмите **Publish app** и подтвердите. Статус сменится на **In production**.
3. Удалите старый токен и войдите заново — токен, выданный в режиме Testing, всё равно истечёт:

   ```bash
   rm secrets/token_<channel_id>.json
   python -m pipeline upload scripts/my_video.json
   ```

Для личного использования **проверка Google не нужна**: отправлять приложение на verification не надо, даже
если консоль это предлагает. Цена — экран «приложение не проверено» при входе: **Advanced → Go to …**
(«Дополнительно → Перейти…»), один раз на каждый токен. У непроверенного приложения есть лимит в 100
пользователей за всё время — вам нужен один.

Токен может перестать работать и в In production: если отозвать доступ на
<https://myaccount.google.com/permissions>, не пользоваться им полгода или пересоздать OAuth-клиент.
Лечится так же — новым входом.

### Шаг 8. Про аудит YouTube API Services

**Что это.** Проверка YouTube, что ваш API-проект соблюдает правила YouTube API Services. Заявка подаётся через
[YouTube API Services — Audit and Quota Extension Form](https://support.google.com/youtube/contact/yt_api_form):
в форме спрашивают, кто вы, номер проекта Google Cloud и как именно вы используете API.

**Зачем он нужен.** Исторически — чтобы ролики вообще могли быть публичными: с 28 июля 2020 года всё, что
загружалось через API из непроверенного проекта, принудительно оставалось **private**, какую приватность ни
запрашивай. По журналу изменений API это правило **отменено 2 ноября 2026 года**. Сейчас аудит нужен для
повышения квот (по умолчанию проект может загрузить 100 видео в сутки и потратить 10 000 единиц на остальные
вызовы — одному каналу этого хватает с большим запасом).

**Как понять, что вас это коснулось.** После каждой загрузки конвейер спрашивает у YouTube фактическую
приватность ролика. Если она отличается от запрошенной, в логе будет предупреждение в рамке из восклицательных
знаков, а `status` покажет `REQUESTED public!`. Тогда:

- опубликуйте ролик вручную: Studio → Content → ролик → Visibility;
- если ручная публикация недоступна или ситуация повторяется — подайте заявку на аудит по ссылке выше и до
  ответа публикуйте вручную (или загружайте как private и переключайте в Studio).

### Шаг 9. Подтвердите канал по телефону

Откройте <https://www.youtube.com/verify> (или Studio → Settings → Channel → Feature eligibility →
Intermediate features) и введите код из SMS. Без этого у канала:

- **не ставятся свои обложки** — YouTube отклонит обложку; конвейер загрузит ролик, сообщит об ошибке и при
  следующем запуске (после подтверждения) дошлёт только обложку;
- **не загружаются ролики длиннее 15 минут** — конвейер проверяет длительность до загрузки и, если канал не
  подтверждён, останавливается, ничего не отправив.

### Шаг 10. Включите автодубляж в Studio

Автодубляж (автоматические звуковые дорожки на других языках) настраивается **только в Studio** — в YouTube
Data API такого параметра нет:

Studio → **Settings** → **Channel** → **Advanced settings** → раздел «Automatic dubbing» → галочка
**Allow automatic dubbing** → Save. (В более старом интерфейсе пункт был в Settings → Upload defaults →
Advanced settings.) Там же можно включить ручную проверку дубляжа перед публикацией. Функция доступна не всем
каналам — если раздела нет, YouTube её вам ещё не открыл.

От конвейера автодубляжу нужно одно: язык оригинальной озвучки. Он передаётся при загрузке
(`defaultAudioLanguage` из поля `language` сценария), а после загрузки проверяется; если YouTube его не
сохранил, в логе будет предупреждение — тогда выставьте «Video language» в карточке ролика вручную.

### После загрузки

- **Связанное видео у Shorts.** Привязать Short к длинному ролику («Related video») через API нельзя — такого
  поля нет. Конвейер ставит ссылку на длинный ролик в описание Short и в конце печатает напоминание со
  ссылками вида `https://studio.youtube.com/video/<id>/edit`: откройте каждую, найдите «Related video» и
  выберите длинный ролик.
- **Квоты.** Загрузки считаются отдельно: 100 в сутки. Остальное — из 10 000 единиц в сутки: обложка ≈ 50,
  субтитры 400, проверки по 1. Счётчики обнуляются в полночь по тихоокеанскому времени. Если квота кончилась,
  команда скажет, какая именно, — запустите её снова на следующий день.

### Несколько каналов

Токен хранится отдельно для каждого `channel_id` из сценария: `secrets/token_<channel_id>.json`. Сценарий с
другим `channel_id` при первой загрузке откроет браузер ещё раз — выберите в нём второй канал. OAuth-клиент
(`client_secret.json`) общий; не забудьте добавить второй аккаунт в test users, если приложение ещё в Testing.

Чтобы исключить ошибку при первом входе, можно заранее записать в `.env`, какому YouTube-каналу соответствует
`channel_id`. ID канала начинается с `UC`; он виден на <https://www.youtube.com/account_advanced>, и его же
конвейер печатает при первом входе:

```
YOUTUBE_CHANNEL_ID__YT_TECH_DISASTERS_EN=UCxxxxxxxxxxxxxxxxxxxxxx
```

Имя переменной — `YOUTUBE_CHANNEL_ID__` плюс `channel_id` заглавными буквами, всё, кроме букв и цифр, заменено
на `_`. Вход в другой канал будет отклонён, токен не сохранится.

### Сервер без браузера

Первый вход требует браузера на той же машине. Войдите один раз на своём компьютере и скопируйте на сервер
`secrets/client_secret.json` и `secrets/token_<channel_id>.json` (это ключи от канала — передавайте по scp, не
по почте). Для Docker — подключите папку: `-v "$PWD/secrets:/app/secrets"`.

## 6. Повторные запуски, сбои, пересборка

- **Повторный запуск пропускает готовое.** У каждой сцены и каждого артефакта есть хеш входных данных;
  пересобирается только то, у чего вход изменился. Поправили текст одной сцены — заново озвучится только она,
  перерендерятся только затронутые клипы.
- **После падения** (сеть, лимиты, Ctrl+C) запустите ту же команду — работа продолжится с места ошибки.
  Причина последней ошибки видна в `python -m pipeline status <video_id>` и в `pipeline.log`.
- **`--force`** пересобирает всё для выбранных этапов. Оплаченные ответы API при этом берутся из `cache/`.
  Чтобы заново запросить (и оплатить) озвучку и картинки, добавьте `--no-cache`.
- На 429 и 5xx запросы повторяются с экспоненциальной задержкой (до 6 попыток).

## 7. Настройки

Все необязательные настройки — в `.env.example` (модель и размер картинок, формат аудио, сглаживание
Ken Burns, число параллельных рендеров, уровень музыки, пути к секретам YouTube, размер куска загрузки). Корень проекта по умолчанию — текущая папка;
другой можно задать через `--root` или `PIPELINE_ROOT`.

## 8. Устройство

| Модуль | Что делает |
|---|---|
| `contract` | pydantic-модели сценария поверх приложенного `validate_script.py` |
| `state` | state-файл, хеши, атомарная запись |
| `tts` | ElevenLabs with-timestamps, mock, кеш, расходы |
| `align` | символы → слова; выравнивание озвучки и субтитров (числа через num2words) |
| `timeline` | покадровый таймлайн сцен и пауз |
| `images` | `ImageProvider`: mock, gpt_image, flux; сборка промпта, кеш, параллелизм |
| `diagrams` | шаблоны big_number, counter, icon_grid, timeline, bar_compare (Pillow) |
| `audio` | стемы, ducking, loudnorm |
| `render` | клипы сцен, переходы, Ken Burns, склейка, проверка синхронизации |
| `captions` | SRT и ASS |
| `metadata` | главы, источники, `metadata.json` |
| `shorts` | вертикальные нарезки |
| `thumbnail` | обложка: картинка через `ImageProvider`, текст через Pillow, подбор качества JPEG |
| `youtube_auth` | OAuth 2.0 (Desktop app), токен на каждый `channel_id`, сверка канала |
| `youtube_api` | пять вызовов YouTube Data API по HTTP, resumable upload, понятные ошибки |
| `upload` | команда `upload`: предусловия, план, идемпотентность, Shorts, проверка после загрузки |
| `cli` | команды `validate`, `run`, `upload`, `status` |

Принятые решения и их причины — в `DECISIONS.md`.

## 9. Частые проблемы

- **`'ffmpeg' not found in PATH`** — FFmpeg не установлен или не виден из venv (см. п. 1).
- **`refused: script status is …`** — сценарий не утверждён. Для пробного прогона используйте `--mock`.
- **`ELEVENLABS_API_KEY is not set`** — нет `.env` или он лежит не в корне проекта.
- **`HTTP 401 … check the API key`** — ключ неверный или отозван.
- **`image provider '…' is not implemented`** — в сценарии `nano_banana` или `ideogram`; задайте
  `--image-provider flux` или `gpt_image`.
- **Субтитры не тем шрифтом** — положите TTF в `assets/fonts/` (п. 3.4).
- **Рендер медленный** — уменьшите `PIPELINE_SUPERSAMPLE` до 2–3 (движение камеры станет чуть менее плавным)
  или увеличьте `PIPELINE_RENDER_WORKERS` на машине с большим числом ядер.

**Загрузка на YouTube**

- **`refused:` и список причин** — не выполнены предусловия (сценарий не утверждён, этап не завершён, нет файла,
  сценарий изменён после рендера, последний прогон был `--mock`). Исправьте и запустите снова.
- **`OAuth client file not found`** — нет `secrets/client_secret.json` (п. 5, шаги 1–4).
- **`Google login failed (access_denied)`** — вход отменён или аккаунта нет в test users (п. 5, шаг 3).
- **`Google rejected the saved token … invalid_grant`** — токен истёк или отозван. Чаще всего это 7 дней режима
  Testing (п. 5, шаг 7). Токен уже удалён — просто запустите команду ещё раз и войдите.
- **`… belongs to YouTube channel …, but Google now reports channel …`** — токен от другого канала. Команда
  ничего не загрузила. Удалите указанный файл токена и войдите в нужный канал.
- **`the daily UPLOAD quota … is used up`** — исчерпаны загрузки (100 в сутки на проект).
  **`the daily YouTube Data API quota … is used up`** — исчерпаны 10 000 единиц на остальные вызовы.
  В обоих случаях запустите ту же команду на следующий день: загруженное пропускается.
- **`YouTube refused the custom thumbnail`** — канал не подтверждён по телефону (п. 5, шаг 9).
- **`… is not allowed to upload videos longer than 15 minutes`** — то же самое; ничего не было отправлено.
- **`YouTube says this channel has reached its upload limit`** — это лимит самого канала, не API. Подождите сутки.
- **`YouTube Data API v3 is not enabled`** — п. 5, шаг 2.
- **Ролик оказался private, хотя запрошен public** — п. 5, шаг 8.
- **`the Google auth libraries are not installed`** — `pip install -r requirements.txt` в активированном venv.
