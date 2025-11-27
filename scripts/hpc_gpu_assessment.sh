#!/bin/bash
echo "=== HPC GPU Resource Assessment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""
echo "SLURM node GPU status:"
sinfo -N -o "%N %G %t" || true
echo ""
echo "Detailed node GPU information:"
scontrol show node $(hostname) | grep -i gpu || true
echo ""
echo "NVIDIA GPU details:"
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
else
  echo "nvidia-smi not available"
fi
echo ""
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
