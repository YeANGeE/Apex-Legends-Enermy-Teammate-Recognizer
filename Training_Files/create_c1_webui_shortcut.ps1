$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$desktopPath = [Environment]::GetFolderPath('Desktop')
$pythonPath = Join-Path $projectRoot '.venv-yolo\Scripts\pythonw.exe'
$launcherPath = Join-Path $projectRoot 'run_c1_webui.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Missing Python: $pythonPath" }
if (-not (Test-Path -LiteralPath $launcherPath)) { throw "Missing launcher: $launcherPath" }
$shortcutPath = Join-Path $desktopPath 'YOLO 敌人检测.lnk'
$shellObject = New-Object -ComObject WScript.Shell
$shortcut = $shellObject.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonPath
$shortcut.Arguments = '"' + $launcherPath + '"'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'YOLO26n Day4 补充训练版 · 图片及逐帧视频检测 · 导出带框 MP4'
$shortcut.IconLocation = "$pythonPath,0"
$shortcut.Save()
$verified = $shellObject.CreateShortcut($shortcutPath)
if ($verified.TargetPath -ne $pythonPath -or $verified.Arguments -ne $shortcut.Arguments) {
    throw 'Shortcut verification failed.'
}
Write-Output "Created and verified: $shortcutPath"
