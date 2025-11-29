sudo docker run \
  --runtime=nvidia \
  --net=host \
  -v $(pwd)/mb-enhance/runtime-cache:/workspace/runtime-cache \
  -v $(pwd)/mb-enhance/runtime-log:/workspace/runtime-log \
  -it cpf-mb-enhance:v0.01