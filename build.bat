@echo off
echo Building CashVision...

REM Set JAVA_HOME to Java 11 if available
if exist "C:\Program Files\Java\jdk-11" (
    set JAVA_HOME=C:\Program Files\Java\jdk-11
) else if exist "C:\Program Files\OpenJDK\jdk-11" (
    set JAVA_HOME=C:\Program Files\OpenJDK\jdk-11
) else (
    echo Warning: Java 11 not found, using default Java
)

REM Build with reduced memory
gradlew.bat assembleDebug --no-daemon --max-workers=1

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo APK location: app\build\outputs\apk\debug\app-debug.apk
) else (
    echo Build failed!
)

pause
