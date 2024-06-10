export USERID=$(id -u)
export GROUPID=$(id -g)
export USERNAME=$(whoami)
export CONTAINERNAME=${1:-idslab_reparam_jetson}
export COMPOSE_PROJECT_NAME=${2:-idslab_reparam_jetson}
#export CONTAINERNAME=${CONTAINERNAME:-pdh_bq_pt_2210}
printf "USERID=%s\n" $USERID 
printf "GROUPID=%s\n" $GROUPID 
printf "USERNAME=%s\n" $USERNAME
printf "CONTAINERNAME=%s\n" $CONTAINERNAME 
xhost +
xhost +local:docker
echo $xhost
docker compose down
docker compose up -d --build --remove-orphans
docker compose exec echo $xhost
docker exec -it $CONTAINERNAME /bin/bash