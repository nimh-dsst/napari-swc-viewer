@echo off
set "REPO_DIR=%USERPROFILE%\repos\napari-swc-viewer"

cd /d "%REPO_DIR%"
if errorlevel 1 (
    echo Could not find the repository at:
    echo   %REPO_DIR%
    echo.
    echo Clone the repository to %%USERPROFILE%%\repos\napari-swc-viewer, or edit this script to use your actual path.
    echo.
    pause
    exit /b 1
)

pixi run napari
set "STATUS=%ERRORLEVEL%"

echo.
if "%STATUS%"=="0" (
    echo napari has closed.
) else (
    echo pixi run napari exited with status %STATUS%.
)

pause
exit /b %STATUS%
