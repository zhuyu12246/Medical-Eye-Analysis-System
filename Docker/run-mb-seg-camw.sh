sudo docker run \
  --runtime=nvidia \
  --net=host \
  -v $(pwd)/mb-seg-camw/runtime-cache:/workspace/runtime-cache \
  -v $(pwd)/mb-seg-camw/runtime-log:/workspace/runtime-log \
  -it cpf-mb-seg-camw:v0.02