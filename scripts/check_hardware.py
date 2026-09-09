"""Record capability detection. Never substitute CPU timings for GPU timings."""
import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path


def probe() -> dict:
    info={'platform':platform.platform(),'python':platform.python_version(),
          'nvidia_smi_found':bool(shutil.which('nvidia-smi')),'torch_installed':False,
          'cuda_available':False,'devices':[], 'hardware_measurements_performed':False}
    if info['nvidia_smi_found']:
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                                  capture_output=True,text=True,timeout=10,check=False)
            info['nvidia_smi_exit_code']=result.returncode
            info['nvidia_smi_devices']=result.stdout.strip().splitlines()
        except (OSError,subprocess.TimeoutExpired): info['nvidia_smi_error']='probe failed'
    if importlib.util.find_spec('torch'):
        import torch
        info.update(torch_installed=True,torch_version=torch.__version__,cuda_available=torch.cuda.is_available())
        if info['cuda_available']:
            info['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info['status']=('CUDA detected; separate measurement procedure required.' if info['cuda_available']
                    else 'Hardware calibration not executed in this environment. No usable CUDA device detected.')
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=probe();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
