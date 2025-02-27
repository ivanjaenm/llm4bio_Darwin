# run this command after updating Dockerfile
#docker login
docker build -f llm4bio.dockerfile -t ivanjaenm/llm4bio .
docker push ivanjaenm/llm4bio
#docker system prune --all --force --volumes
