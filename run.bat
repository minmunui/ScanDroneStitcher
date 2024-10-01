@echo off

SETLOCAL ENABLEDELAYEDEXPANSION

set "base_folder=./data/input/rotated"

for /D %%d in ("%base_folder%/*") do (
set "target_folder=./data/input/rotated/%%~nxd"
echo Target folder: !target_folder!
python .\src\original_cv.py !target_folder! --work_megapix 0.6 --matcher affine --estimator affine  --ba affine --ba_refine_mask xxxxx --wave_correct no --warp affine --output def_%%d.jpg
python .\src\original_cv.py !target_folder! --work_megapix 0.6 --output no_%%d.jpg
)