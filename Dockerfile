# Pin used for the release validation; update only with a new validation receipt.
FROM python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /opt/visual-repair-lab
COPY . .
RUN python -m pip install --no-cache-dir -r requirements-test.lock . && python -m pip check
RUN mkdir -p /data && chown 65532:65532 /data
USER 65532:65532
ENTRYPOINT ["visual-repair-lab"]
CMD ["serve", "--bank", "banks/train.json", "--database", "/data/episodes.db", "--host", "0.0.0.0", "--port", "8822"]
