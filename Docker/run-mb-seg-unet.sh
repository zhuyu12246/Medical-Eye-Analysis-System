sudo docker run \
  --runtime=nvidia \
  --net=host \
  -v $(pwd)/mb-seg-unet/runtime-cache:/workspace/runtime-cache \
  -v $(pwd)/mb-seg-unet/runtime-log:/workspace/runtime-log \
  -it cpf-mb-seg-unet:v0.02