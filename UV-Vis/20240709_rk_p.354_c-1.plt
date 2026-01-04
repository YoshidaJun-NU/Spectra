## UVplot csvファイル
#  


#好み フォント + 大きさ
#set terminal wxt font "arial,12"

#set terminal qt font "arial,30"

#CSVファイルの場合
set datafile separator ","

#表示範囲
set xrange [250:600]
set yrange [0:1.5]

#色変化の選択
#et palette model RGB functions gray,0,0 # HSV color space
set palette model RGB rgbformulae 7,5,15  # traditional pm3d (black-blue-red-yellow)
#set palette model RGB rgbformulae 35,13,10 #rainbow (blue-green-yellow-red)

set palette model RGB rgbformulae 30,31,32 # color printable on gray (black-blue-violet-yellow-white)

unset colorbox
#n/10　の部分をn/5にすると変化幅は大きくなる
#n+1をn+2でも変化幅は大きくなる
#n/10が1を超えるとおかしくなる。

plot n=0, '20240709_rk_p.354_c-1 0分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "0 min"

replot n=n+1, '20240709_rk_p.354_c-1 5分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "5 min"

replot n=n+1, '20240709_rk_p.354_c-1 10分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "10 min"

replot n=n+1, '20240709_rk_p.354_c-1 15分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "15 min"

replot n=n+1, '20240709_rk_p.354_c-1 20分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "20 min"

replot n=n+1, '20240709_rk_p.354_c-1 25分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "25 min"

replot n=n+1, '20240709_rk_p.354_c-1 30分.csv' every ::19 u 1:2 with lines lc palette frac n/10.0 t "30 min"


 

#軸
set xlabel '{/Arial=12 Wavelength} {/Arial-italic=12 / nm}'
set ylabel '{/Arial=12 Abs}'
#set ylabel '{/Symbol:Italic=12 e} {/Arial-italic=12 / M^{-1} cm^{-1}}'
set xtics 100
set ytics 0.5
set tics out

set xzeroaxis lt 0
unset x2zeroaxis
unset y2zeroaxis

set ytics nomirror
set xtics nomirror



#凡例の位置 left, right, top, bottom, outside, below
#set key outside
#凡例を非表示の場合
unset key

#set terminal windows
replot
#set output


#色
#medium-blue, dark-red, light-red, goldenrod, dark-turquoise, dark-violet


#    EOF
