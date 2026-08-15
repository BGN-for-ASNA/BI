# run_beast_fixed.ps1
$beast_home = "C:\Users\Sosa\Downloads\BEAST.v2.7.7.Windows\BEAST"
$beast_bat = "$beast_home\bat\beast.bat"
$pkg_dir = "$beast_home\lib\packages"

Write-Host "Running Model 1 (Spatial Heterogeneity)..." -ForegroundColor Cyan
& $beast_bat -packagedir $pkg_dir -overwrite -threads 4 "C:\Users\Sosa\Documents\BF\Test\Modeling Evolutionary Heterogeneity (WIP)\Model_1_Spatial_Heterogeneity\beast_model1_gamma.xml"

Write-Host "`nRunning Model 2 (Temporal Heterogeneity - UCLN)..." -ForegroundColor Cyan
& $beast_bat -packagedir $pkg_dir -overwrite -threads 4 "C:\Users\Sosa\Documents\BF\Test\Modeling Evolutionary Heterogeneity (WIP)\Model_2_Temporal_Heterogeneity\beast_model2_ucln.xml"

Write-Host "`nGenerating comparisons..." -ForegroundColor Green
python "C:\Users\Sosa\Documents\BF\Test\Modeling Evolutionary Heterogeneity (WIP)\compare_beast_BF.py"
