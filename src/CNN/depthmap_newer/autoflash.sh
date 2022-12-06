make -r clean
make -r
C:\MaximSDK\Tools\OpenOCD\openocd.exe -s C:/MaximSDK/Tools/OpenOCD/scripts -f interface/cmsis-dap.cfg -f target/max78000.cfg  -c "program build/depthmap.elf reset exit"

