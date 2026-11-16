@echo off
setlocal
cd /d "%~dp0"
echo.
echo === More than a Hack 2027: jalankan semuanya ===
echo.

rem 1. Python ada?
python --version >nul 2>&1
if errorlevel 1 (
  echo GAGAL: Python tidak ditemukan. Pasang dari python.org dan centang Add Python to PATH.
  pause
  exit /b 1
)

rem 2. Data ada? Pakai MTAH_DATA_DIR kalau diset, kalau tidak folder data di sini.
set "DATA=%MTAH_DATA_DIR%"
if "%DATA%"=="" set "DATA=%~dp0data"
if not exist "%DATA%\tactical\BAR-ATH_tactical_atd.txt" (
  echo GAGAL: file data tidak ditemukan di "%DATA%\tactical\"
  echo Taruh data pertandingan di folder data\tactical\ sesuai data\README.md, lalu jalankan lagi.
  pause
  exit /b 1
)

rem 3. Paket
echo [1/3] memasang paket...
python -m pip install -r requirements-solution.txt
if errorlevel 1 (
  echo Versi terkunci gagal dipasang, mencoba versi terbaru yang cocok dengan Python Anda...
  python -m pip install numpy pandas
  if errorlevel 1 (
    echo GAGAL memasang numpy dan pandas.
    pause
    exit /b 1
  )
)

rem 4. Uji
echo.
echo [2/3] menjalankan uji...
python tests\test_possession_hmm.py
if errorlevel 1 (
  echo GAGAL: ada uji yang tidak lulus. Kirim pesan errornya, jangan lanjut mengumpulkan.
  pause
  exit /b 1
)

rem 5. Buat dan validasi file pengiriman
echo.
set /p TEAM=Nama tim persis seperti di formulir pendaftaran:
echo.
echo [3/3] membuat dan memvalidasi file pengiriman...
python solution\make_submission.py --team "%TEAM%"
if errorlevel 1 (
  echo GAGAL: file TIDAK boleh diunggah. Kirim pesan errornya.
  pause
  exit /b 1
)

echo.
echo SELESAI. File ada di folder submission\
echo Unggah persis file itu. Salin angka ringkasan di atas, bukan isi datanya.
pause
