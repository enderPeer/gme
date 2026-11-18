# Loads the cluster host configuration: cluster/hosts.local.ps1 if present (your
# machines, git-ignored), otherwise the placeholders in cluster/hosts.example.ps1.
$gmeHostsLocal = Join-Path $PSScriptRoot 'hosts.local.ps1'
if (Test-Path $gmeHostsLocal) {
    . $gmeHostsLocal
} else {
    . (Join-Path $PSScriptRoot 'hosts.example.ps1')
    Write-Warning 'cluster/hosts.local.ps1 not found: using the placeholder hosts of cluster/hosts.example.ps1'
}
