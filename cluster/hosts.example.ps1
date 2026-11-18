# Cluster hosts for the GME scripts (cluster/*.ps1).
# Copy this file to cluster/hosts.local.ps1 (git-ignored) and fill in your machines.
# Rigs need SSH key login, the NVIDIA driver, gcc, cmake, tmux and an NVRTC library
# (e.g. pip install nvidia-cuda-nvrtc in a venv; see cluster/*.sh). No CUDA toolkit needed.
$GmeUser = 'user'           # SSH user on the rigs
$GmeRigA = 'rig-a.local'    # rig A: two large GPUs (CUDA devices 0 and 1)
$GmeRigB = 'rig-b.local'    # rig B: three smaller GPUs (CUDA devices 0, 1, 2)
