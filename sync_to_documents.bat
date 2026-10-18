@echo off
set SRC=guide_packets
set DST=c:\Users\sridhar\OneDrive\Documents\PrismDock_Guide_Packets

echo Syncing %SRC% to %DST%...
xcopy /E /I /Y /Q "%SRC%\*" "%DST%\"
echo Sync completed successfully!
