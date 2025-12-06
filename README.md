# shallenge_cl
OpenCL + C++ = https://shallenge.quirino.net/

## How to use (vast.ai)

```shell
export SSH_USERNAME="root"
export SSH_PORT="40708"
export SSH_HOST="108.236.147.253"

# sync files
rsync -avz --exclude='.git' --exclude='.DS_Store' -e "ssh -p $SSH_PORT" ~/Desktop/shallenge_cl/ $SSH_USERNAME@$SSH_HOST:~/shallenge_cl

# open shell
ssh -p $SSH_PORT $SSH_USERNAME@$SSH_HOST

# install dependencies
apt-get install -y xxd

# build project
cd shallenge_cl
make DEFAULT_USERNAME=someone
./output/challenge_cl
```
