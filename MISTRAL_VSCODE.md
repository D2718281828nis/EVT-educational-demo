# Запуск Graph-EVT-agent с Mistral из VS Code

## Главное ограничение GitHub Secrets

Секрет `MISTRAL_API_KEY`, сохранённый в настройках GitHub, нельзя прочитать из
локального процесса VS Code. GitHub намеренно не показывает значение секрета
после сохранения и подставляет его только в окружение job, запущенного через
GitHub Actions.

Кроме того, repository secret принадлежит конкретному репозиторию. Workflow из
`EVT-educational-demo` не получит secret, который сохранён только в
`Graph-EVT-agent`. Для запуска есть два безопасных варианта:

1. **Локальная разработка в VS Code:** сохранить ключ локально в переменной
   окружения или в игнорируемом файле `.env`.
2. **Запуск с уже сохранённым GitHub secret:** запускать программу в GitHub
   Actions того репозитория, которому доступен secret (либо добавить такой же
   secret в `EVT-educational-demo`). Это удалённый запуск, а не запуск на машине
   VS Code.

## Вариант 1: локальный запуск из VS Code

### 1. Использовать существующий VENV

Если VENV уже создан, повторно выполнять `python -m venv` не нужно. Откройте
корень `EVT-educational-demo` в VS Code, выберите команду **Python: Select
Interpreter** и укажите Python именно из своего VENV:

- Linux/macOS: `<путь-к-VENV>/bin/python`;
- Windows: `<путь-к-VENV>\Scripts\python.exe`.

Затем активируйте тот же VENV во встроенном терминале. Если он находится в
`.venv` внутри проекта:

```bash
source .venv/bin/activate
```

Для Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Убедитесь, что команды `python` и `pip` относятся к выбранному VENV:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Путь в обоих результатах должен вести в ваш VENV. Далее устанавливайте
`Graph-EVT-agent` **в этот активированный VENV** способом, указанным в README
библиотеки. Если репозиторий оформлен как устанавливаемый Python-пакет, обычно
подходит:

```bash
python -m pip install "git+https://github.com/D2718281828nis/Graph-EVT-agent.git"
```

Если у проекта нет `pyproject.toml`/`setup.py`, клонируйте его рядом и
установите его `requirements.txt`:

```bash
git clone https://github.com/D2718281828nis/Graph-EVT-agent.git ../Graph-EVT-agent
python -m pip install -r ../Graph-EVT-agent/requirements.txt
```

### 2. Передать ключ локально

VENV изолирует Python и установленные пакеты, но сам по себе не предоставляет
`MISTRAL_API_KEY`. Создайте в корне проекта отдельный файл `.env` (он
игнорируется Git):

```dotenv
MISTRAL_API_KEY=ваш_локальный_ключ
```

Не коммитьте `.env`, не вставляйте ключ в `settings.json`, `launch.json`, код,
ноутбук или терминальный вывод. Значение GitHub secret восстановить нельзя:
если локальной копии нет, создайте отдельный ключ Mistral для разработки.

### 3. Настроить отладчик

Создайте `.vscode/launch.json` локально (либо адаптируйте существующий):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "EVT agent",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/path/to/your_entrypoint.py",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal"
    }
  ]
}
```

Замените `path/to/your_entrypoint.py` на файл, который создаёт/запускает агента
согласно API `Graph-EVT-agent`. Отладчик использует ранее выбранный
интерпретатор VENV, а `envFile` передаёт ему `MISTRAL_API_KEY`. Запустите
конфигурацию клавишей **F5**.

Для запуска без отладчика в Linux/macOS сначала активируйте VENV, а затем
экспортируйте переменные из `.env`:

```bash
set -a
source .env
set +a
python path/to/your_entrypoint.py
```

В PowerShell файл `.env` автоматически не загружается. Можно безопасно
прочитать конкретную переменную без вывода её значения:

```powershell
.\.venv\Scripts\Activate.ps1
$env:MISTRAL_API_KEY = (Get-Content .env |
  Where-Object { $_ -match '^MISTRAL_API_KEY=' } |
  Select-Object -First 1) -replace '^MISTRAL_API_KEY=', ''
python path\to\your_entrypoint.py
```

Код должен получать ключ из окружения, например:

```python
import os

mistral_api_key = os.environ["MISTRAL_API_KEY"]
```

Если библиотека сама читает `MISTRAL_API_KEY`, явно передавать значение в её
конструктор не нужно.

## Вариант 2: запуск через GitHub Actions

Если добавить `MISTRAL_API_KEY` в **Settings → Secrets and variables → Actions**
репозитория `EVT-educational-demo`, минимальная часть workflow выглядит так:

```yaml
name: Run EVT agent

on:
  workflow_dispatch:

jobs:
  run-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install "git+https://github.com/D2718281828nis/Graph-EVT-agent.git"
      - run: python path/to/your_entrypoint.py
        env:
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
```

Workflow можно запустить в VS Code через расширение **GitHub Actions**, но код
всё равно исполнится на GitHub runner. Для private-репозитория зависимости
потребуется отдельная аутентификация; `MISTRAL_API_KEY` для клонирования
использовать нельзя.

Если ключ должен оставаться только в `Graph-EVT-agent`, workflow необходимо
разместить там (например, checkout обоих репозиториев) либо оформить reusable
workflow и явно передавать ему secret из вызывающего репозитория, которому этот
secret доступен. Секреты не наследуются произвольно между репозиториями и не
передаются workflow из fork/pull request от недоверенного автора.

## Быстрая диагностика без раскрытия ключа

Проверить только наличие переменной можно так:

```bash
python -c 'import os; assert os.getenv("MISTRAL_API_KEY"), "MISTRAL_API_KEY is not set"; print("MISTRAL_API_KEY is set")'
```

Никогда не используйте `echo $MISTRAL_API_KEY` и не печатайте объект
конфигурации клиента целиком.
