#!/bin/bash

export HOST=$AIP_HTTP_HOST
export PORT=$AIP_HTTP_PORT
export WORKER_THREADS=$WORKER_THREADS

uvicorn app.main:app --host=$HOST --port=$PORT --workers=$WORKER_THREADS
