#!/bin/bash

set -xe

# Build and run
gcc -Wall -Wextra -O2 -o main main.c -lm
rm -rf output output_padded demo.gif
mkdir -p output
./main

# Rename to zero-padded format for ffmpeg
mkdir -p output_padded
for f in output/weight_*.ppm; do
    num=$(echo "$f" | grep -oP 'weight_\K\d+')
    printf -v padded "%03d" "$num"
    cp "$f" "output_padded/weight_${padded}.ppm"
done

# Generate GIF
ffmpeg -y -i output_padded/weight_%03d.ppm output/demo.mp4
ffmpeg -y -i output/demo.mp4 -vf "fps=30,scale=500:500:flags=lanczos,palettegen" output/palette.png
ffmpeg -y -i output/demo.mp4 -i output/palette.png -filter_complex "fps=10,scale=500:500:flags=lanczos,paletteuse" demo.gif

# Cleanup
rm -rf output_padded output/demo.mp4 output/palette.png

echo "Done: demo.gif"
