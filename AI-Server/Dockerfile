FROM continuumio/miniconda 

# Create work directory
WORKDIR /app/ai-server

COPY . .

# Create conda environment
RUN conda env create -f env.yml

# Activate conda environment
ENV PATH /opt/conda/envs/ai-server/bin:$PATH
RUN /bin/bash -c "source activate ai-server"

# Delete some file don't need 
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete    
# Run server
CMD ["python", "temp-predict.py"]
