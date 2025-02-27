# run this command after updating Dockerfile
#docker login
docker build -t ivanjaenm/llm4bio:2.0 .
docker push ivanjaenm/llm4bio:2.0
#docker system prune --all --force --volumes
