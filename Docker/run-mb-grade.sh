sudo docker run \
  --runtime=nvidia \
  --net=host \
  -v $(pwd)/mb-seg-grade/runtime-cache:/workspace/runtime-cache \
  -v $(pwd)/mb-seg-grade/runtime-log:/workspace/runtime-log \
  -it cpf-mb-grade:v0.01