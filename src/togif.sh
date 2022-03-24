echo $1
./main x.mp4 0.001 0.001 0.005 $1
ffmpeg -i x.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 ../contents/out$1.gif
mv x.mp4.png ../contents/init_out$1.png
rm -rf x.mp4 x.mp4.png
