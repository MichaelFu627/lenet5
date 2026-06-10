FROM python:3.11-slim

# HF Spaces 推荐 non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860

CMD ["python", "-m", "backend.api.server", "--port", "7860", "--host", "0.0.0.0"]