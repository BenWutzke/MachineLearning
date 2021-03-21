FROM python:3.6.6-slim

# Mount current directory to /app in the container image
#VOLUME ./:app/

# Copy local directory to /app in container
# Dont use COPY * /app/ , * will lead to lose of folder structure in /app
COPY ../src/app/. /app/

# Change WORKDIR
WORKDIR /app

# Install dependencies
# use --proxy http://<proxy host>:port if you have proxy
RUN pip install -r ../src/artifacts/requirements.txt

# In Docker, the containers themselves can have applications running on ports. To access these applications, we need to expose the containers internal port and bind the exposed port to a specified port on the host.
# Expose port and run the application when the container is started
EXPOSE 9999:9999
ENTRYPOINT python main.py 9999