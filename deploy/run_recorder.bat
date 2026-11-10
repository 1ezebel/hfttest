@echo off
rem Автоперезапуск рекордера на Windows (ноутбук). Запускать из папки hft-lab:
rem     deploy\run_recorder.bat
rem Чтобы стартовал сам: Планировщик заданий -> Создать задачу -> Триггер "При входе в систему".
rem Отключите сон ноутбука, иначе в записи будут дыры.
set PYTHONPATH=src
:loop
.venv\Scripts\python -m hftlab.recorder --out data
echo Рекордер остановился, перезапуск через 5 секунд (Ctrl+C чтобы выйти)
timeout /t 5
goto loop
