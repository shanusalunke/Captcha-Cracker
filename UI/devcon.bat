echo "Now working"
@echo off
set path="C:\Python27"
cd C:\Python27\Captcha_Cracker\Algorithms
echo Devcon: Calling Python Script Entered as Parameters
echo %ERRORLEVEL%
echo %1 %2 %3 %4
python %1 %2 %3 %4
echo %ERRORLEVEL%
pause