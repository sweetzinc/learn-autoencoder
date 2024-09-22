
## devcontainer
Note: a local (Windows11) folder is mounted to the devcontainer via `devcontainer.json: "mounts" `

1. VSCode Command: Dev Containers: Open Folder in Container...
2. VSCode Terminal: `python torch_gpu_test.py`


### Docker test
```bash
docker build -t ae:dev ./.devcontainer --no-cache 
```

