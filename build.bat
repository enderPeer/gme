@echo off
rem Configure + build GME with MSVC (host compiler for nvcc) and Ninja.
rem   build.bat [build-dir] [target]     (defaults: build, all targets)
setlocal
set "BDIR=%~1"
if "%BDIR%"=="" set "BDIR=build"
set "TARGET=%~2"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
  echo Visual Studio C++ build tools not found.
  exit /b 1
)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if not exist "%BDIR%\build.ninja" (
  cmake -S . -B "%BDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl || exit /b 1
)
if "%TARGET%"=="" (
  cmake --build "%BDIR%" || exit /b 1
) else (
  cmake --build "%BDIR%" --target %TARGET% || exit /b 1
)
echo.
echo Built %BDIR% %TARGET%
