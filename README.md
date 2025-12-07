# shallenge_cl
OpenCL + C++ = https://shallenge.quirino.net/

## How to use (vast.ai)

```shell
export SSH_USERNAME="root"
export SSH_PORT="34088"
export SSH_HOST="198.53.64.194"

# sync files
rsync -avz --exclude='.git' --exclude='.DS_Store' -e "ssh -p $SSH_PORT" . $SSH_USERNAME@$SSH_HOST:/workspace/shallenge_cl

# open shell
ssh -p $SSH_PORT $SSH_USERNAME@$SSH_HOST

# install dependencies
apt-get install -y xxd

# build project
cd shallenge_cl
make DEFAULT_USERNAME=brandonros GLOBAL_SIZE=2097152 LOCAL_SIZE=256 HASHES_PER_THREAD=256
./output/shallenge_cl
```
