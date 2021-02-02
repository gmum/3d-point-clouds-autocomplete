## Requirements
- dependencies stored in `requirements.txt`.
- Python 3.7+
- cuda

## Installation


Create and activate your conda env:
- run `conda create --name <your env name> python=3.7`
- run `conda activate <your env name>`

Install requirements:
- run `conda install pytorch torchvision torchaudio cudatoolkit=<your CUDA version (ex. 10.2)> -c pytorch`
- run 'pip install -r requirements.txt' (brakuje niektórych pakietów, w razie problemów napisz w Teamsach (Artur Kasymov) )

Then execute:
```
export CUDA_HOME=... # e.g. /var/lib/cuda-10.0/
./build_losses.sh
```

## Usage
**Add project root directory to PYTHONPATH**

```export PYTHONPATH=project_path:$PYTHONPATH```

1) make a copy of the sample configs:
- run `cp setting/config.json.sample setting/config.json`

2) specify your personal configs in `setting/config.json`:
- change ["dataset"]["path"] and ["results_root"] field 
- select your GPU id for in ["setup"]["gpu_id"]
- select the batch_size for your device in ["training"]["dataloader"]

3) exec script
- run `python3 core/main.py --config settings/config.json`



## Extending
In case you want create your own experiments: 
1) write you experiment function in core/exps
2) add it to `experiment_functions_dict` in core/exps
3) include your special parameters in config.sample.json ["experiments]["<your func name>"] (be sure to add a bool field "execute" there)  
