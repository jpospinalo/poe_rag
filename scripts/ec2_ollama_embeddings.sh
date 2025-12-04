sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl ca-certificates

curl -fsSL https://ollama.com/install.sh | sh

sudo systemctl enable ollama
sudo systemctl start ollama

ollama pull embeddinggemma:latest
