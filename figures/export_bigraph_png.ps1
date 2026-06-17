Add-Type -AssemblyName System.Drawing

$outPath = Join-Path $PSScriptRoot "bigraph_network_architecture.png"
$w = 1800
$h = 1180
$bmp = New-Object System.Drawing.Bitmap($w, $h)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear([System.Drawing.Color]::White)

function Pen($hex, $width=2) {
    return New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml($hex), $width)
}

function Brush($hex) {
    return New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml($hex))
}

function FontA($size, $style="Regular") {
    return New-Object System.Drawing.Font("Arial", $size, [System.Drawing.FontStyle]::$style, [System.Drawing.GraphicsUnit]::Pixel)
}

function RoundRectPath($x, $y, $w, $h, $r) {
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = 2 * $r
    $p.AddArc($x, $y, $d, $d, 180, 90)
    $p.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $p.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $p.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $p.CloseFigure()
    return $p
}

function DrawBox($x, $y, $w, $h, $fill, $stroke, $dash=$false) {
    $path = RoundRectPath $x $y $w $h 8
    $b = Brush $fill
    $p = Pen $stroke 2
    if ($dash) { $p.DashPattern = @(8, 6) }
    $script:g.FillPath($b, $path)
    $script:g.DrawPath($p, $path)
    $b.Dispose(); $p.Dispose(); $path.Dispose()
}

function DrawTextC($text, $x, $y, $size=16, $style="Regular", $color="#1f2933") {
    $font = FontA $size $style
    $brush = Brush $color
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    $script:g.DrawString($text, $font, $brush, [System.Drawing.PointF]::new($x, $y), $fmt)
    $font.Dispose(); $brush.Dispose(); $fmt.Dispose()
}

function DrawTextL($text, $x, $y, $size=16, $style="Regular", $color="#1f2933") {
    $font = FontA $size $style
    $brush = Brush $color
    $script:g.DrawString($text, $font, $brush, [System.Drawing.PointF]::new($x, $y))
    $font.Dispose(); $brush.Dispose()
}

function DrawArrow($x1, $y1, $x2, $y2, $color="#2f3a45", $width=2.2, $dash=$false) {
    $p = Pen $color $width
    if ($dash) { $p.DashPattern = @(8, 6) }
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(5, 7, $true)
    $p.CustomEndCap = $cap
    $script:g.DrawLine($p, $x1, $y1, $x2, $y2)
    $cap.Dispose(); $p.Dispose()
}

function DrawCurveArrow($points, $color="#2f3a45", $width=2.2, $dash=$false) {
    $p = Pen $color $width
    if ($dash) { $p.DashPattern = @(8, 6) }
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(5, 7, $true)
    $p.CustomEndCap = $cap
    $script:g.DrawCurve($p, $points)
    $cap.Dispose(); $p.Dispose()
}

$inputFill = "#e8f4f8"; $inputStroke = "#2f9eb3"
$embedFill = "#f0f7e8"; $embedStroke = "#5b9b3e"
$layerFill = "#fff4d6"; $layerStroke = "#d99a1e"
$projFill = "#f4ecfb"; $projStroke = "#8b5fbf"
$globalFill = "#ecfdf3"; $globalStroke = "#2f855a"
$actorFill = "#eef2ff"; $actorStroke = "#4c63b6"
$criticFill = "#fff0f0"; $criticStroke = "#cf4f4f"
$maskFill = "#f2f4f7"; $maskStroke = "#77808a"

DrawTextC "BiGraphNetwork Architecture for Flexible Job Shop Scheduling" 900 32 34 Bold
DrawTextC "Operation-machine bipartite graph encoder with pair-wise attention bias, temporal GRU memory, and PPO actor-critic heads" 900 72 18 Regular "#52606d"

DrawTextL "State features" 95 125 20 Bold
DrawBox 70 175 260 78 $inputFill $inputStroke
DrawTextC "Operation nodes" 200 200 18 Bold
DrawTextC "X^O in R^(B x J x 8)" 200 228 15 Regular "#3e4c59"
DrawBox 70 305 260 78 $inputFill $inputStroke
DrawTextC "Machine nodes" 200 330 18 Bold
DrawTextC "X^M in R^(B x M x 5)" 200 358 15 Regular "#3e4c59"
DrawBox 70 435 260 78 $inputFill $inputStroke
DrawTextC "Operation-machine pairs" 200 460 18 Bold
DrawTextC "X^P in R^(B x J x M x 6)" 200 488 15 Regular "#3e4c59"
DrawBox 70 565 260 78 $maskFill $maskStroke $true
DrawTextC "Dynamic pair mask" 200 590 18 Bold
DrawTextC "invalid operation-machine pairs" 200 618 15 Regular "#3e4c59"

DrawTextL "Feature embedding" 420 125 20 Bold
DrawBox 390 175 245 78 $embedFill $embedStroke
DrawTextC "Job MLP" 512 200 18 Bold
DrawTextC "8 -> 128" 512 228 15 Regular "#3e4c59"
DrawBox 390 305 245 78 $embedFill $embedStroke
DrawTextC "Machine MLP" 512 330 18 Bold
DrawTextC "5 -> 128" 512 358 15 Regular "#3e4c59"
DrawBox 390 435 245 78 $embedFill $embedStroke
DrawTextC "Pair MLP" 512 460 18 Bold
DrawTextC "6 -> 128" 512 488 15 Regular "#3e4c59"
DrawArrow 330 214 390 214
DrawArrow 330 344 390 344
DrawArrow 330 474 390 474
DrawArrow 330 604 690 596 "#77808a" 2 $true

DrawBox 700 130 520 610 "#fbfcfd" "#d5dce3"
DrawTextC "Bipartite graph encoder, repeated L = 2 layers" 960 143 20 Bold
DrawBox 760 205 400 112 $layerFill $layerStroke
DrawTextC "O <- M multi-head cross-attention" 960 232 18 Bold
DrawTextC "Q from operations; K,V from machines" 960 262 15 Regular "#3e4c59"
DrawTextC "score = QK^T/sqrt(d_h) + w_a(h_pair)" 960 288 15 Regular "#3e4c59"
DrawBox 760 355 400 112 $layerFill $layerStroke
DrawTextC "M <- O multi-head cross-attention" 960 382 18 Bold
DrawTextC "transposed pair features and mask" 960 412 15 Regular "#3e4c59"
DrawTextC "attention-weighted pair gate: 1 + tanh(W_g h_pair)" 960 438 15 Regular "#3e4c59"
DrawBox 760 505 400 112 $layerFill $layerStroke
DrawTextC "Pair update" 960 532 18 Bold
DrawTextC "concat(h_O, h_M, h_P) -> Linear + tanh" 960 562 15 Regular "#3e4c59"
DrawTextC "residual connection + LayerNorm" 960 588 15 Regular "#3e4c59"
DrawArrow 635 214 760 244
DrawArrow 635 344 760 395
DrawArrow 635 474 760 548
DrawArrow 960 317 960 355 "#5f6b76" 1.7
DrawArrow 960 467 960 505 "#5f6b76" 1.7
DrawBox 805 650 310 62 $embedFill $embedStroke
DrawTextC "Encoded embeddings" 960 672 18 Bold
DrawTextC "h_O, h_M, h_P all with d = 128" 960 697 15 Regular "#3e4c59"

DrawTextL "Policy/value feature paths" 1324 125 20 Bold
DrawBox 1280 185 245 82 $projFill $projStroke
DrawTextC "Local actor projections" 1402 211 18 Bold
DrawTextC "O, M, P: 128 -> 32" 1402 240 15 Regular "#3e4c59"
DrawBox 1280 325 245 96 $globalFill $globalStroke
DrawTextC "Global projections" 1402 352 18 Bold
DrawTextC "O, M, P: 128 -> 16" 1402 381 15 Regular "#3e4c59"
DrawTextC "masked mean pooling" 1402 405 15 Regular "#3e4c59"
DrawBox 1280 475 245 82 $globalFill $globalStroke
DrawTextC "Graph summary" 1402 501 18 Bold
DrawTextC "h_graph = [g_O, g_M, g_P] in R^48" 1402 530 15 Regular "#3e4c59"
DrawBox 1280 610 245 82 $globalFill $globalStroke
DrawTextC "GRUCell memory" 1402 636 18 Bold
DrawTextC "h_hist: 64, input h_graph" 1402 665 15 Regular "#3e4c59"
DrawArrow 1115 681 1280 226
DrawArrow 1115 681 1280 374
DrawArrow 1402 421 1402 475
DrawArrow 1402 557 1402 610

DrawBox 560 825 510 132 $actorFill $actorStroke
DrawTextC "Candidate pair feature construction" 815 853 18 Bold
DrawTextC "for each valid operation-machine pair (O_j, M_k)" 815 884 15 Regular "#3e4c59"
DrawTextC "z_jk = [a_O,j, a_M,k, g_O, g_M, a_P,jk, h_hist] in R^160" 815 916 16 Bold
DrawTextC "32 + 32 + 16 + 16 + 32 + 64" 815 943 13 Regular "#52606d"
DrawBox 1185 825 250 132 $actorFill $actorStroke
DrawTextC "Actor MLP" 1310 853 18 Bold
DrawTextC "160 -> 64 -> 64 -> 1" 1310 884 15 Regular "#3e4c59"
DrawTextC "masked logits + softmax" 1310 916 15 Regular "#3e4c59"
DrawTextC "policy pi(a | s)" 1310 944 16 Bold
DrawBox 1185 1000 250 98 $criticFill $criticStroke
DrawTextC "Critic MLP" 1310 1029 18 Bold
DrawTextC "[h_graph, h_hist] in R^112" 1310 1059 15 Regular "#3e4c59"
DrawTextC "state value V(s)" 1310 1086 16 Bold
DrawArrow 1402 267 970 825 "#2b6cb0"
DrawArrow 1402 692 905 825 "#2b6cb0"
DrawArrow 1280 516 880 825 "#2b6cb0"
DrawArrow 1070 891 1185 891 "#2b6cb0"
DrawArrow 330 604 1185 917 "#77808a" 2 $true
DrawArrow 1402 557 1435 1048
DrawArrow 1402 692 1435 1048

DrawBox 1515 825 220 132 $inputFill $inputStroke
DrawTextC "Scheduling action" 1625 853 18 Bold
DrawTextC "select next pair" 1625 884 15 Regular "#3e4c59"
DrawTextC "a = (operation, machine)" 1625 916 16 Bold
DrawTextC "sample or argmax from pi" 1625 944 15 Regular "#3e4c59"
DrawBox 1515 1000 220 98 $criticFill $criticStroke
DrawTextC "PPO training" 1625 1029 18 Bold
DrawTextC "policy loss + value loss" 1625 1059 15 Regular "#3e4c59"
DrawTextC "entropy regularization" 1625 1086 15 Regular "#3e4c59"
DrawArrow 1435 891 1515 891
DrawArrow 1435 1049 1515 1049
DrawArrow 1625 957 1625 1000 "#5f6b76" 1.7

DrawBox 70 840 390 242 "#fbfcfd" "#d5dce3"
DrawTextL "Notation" 95 858 20 Bold
DrawTextL "O: operation node; M: machine node; P: pair/edge feature" 95 902 15 Regular "#3e4c59"
DrawTextL "J: number of schedulable operations; M: number of machines" 95 932 15 Regular "#3e4c59"
DrawTextL "d = 128; actor local dim = 32; global dim = 16" 95 962 15 Regular "#3e4c59"
DrawTextL "The mask removes infeasible or already invalid pairs" 95 992 15 Regular "#3e4c59"
DrawTextL "Temporal memory is reset for finished environments" 95 1022 15 Regular "#3e4c59"
DrawTextL "Implementation source: model/BiGraphNetwork.py" 95 1055 13 Regular "#52606d"

$bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()
Write-Output $outPath
