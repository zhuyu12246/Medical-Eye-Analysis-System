volume_name="mb-gateway-config"

if sudo docker volume inspect $volume_name >/dev/null 2>&1; then
    echo "has find $volume_name skip"
else
    sudo docker volume create $volume_name
    echo "created $volume_name "
fi

echo "volume location is: "
sudo docker volume inspect $volume_name --format '{{ .Mountpoint }}'


sudo docker run \
  --runtime=nvidia \
  --net=host \
  -v $(pwd)/mb-gateway/runtime-cache:/workspace/runtime-cache \
  -v $(pwd)/mb-gateway/runtime-log:/workspace/runtime-log \
  -v $volume_name:/workspace/Common \
  -it cpf-mb-gateway:v0.02