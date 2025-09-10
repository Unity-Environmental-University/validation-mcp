Param(
  [string]$ServiceName = "validation-mcp",
  [string]$PythonExe = "python",
  [string]$WorkingDir = "$PSScriptRoot\..\",
  [string]$CmdArgs = "-m validation_mcp.mcp_server"
)
$bin = Join-Path $WorkingDir "validation_mcp"
Write-Host "Installing service $ServiceName..."
New-Service -Name $ServiceName -BinaryPathName "`"$PythonExe`" $CmdArgs" -DisplayName $ServiceName -StartupType Automatic -Description "Validation MCP stdio server"
Write-Host "Service installed. Use 'Start-Service $ServiceName' to run."

