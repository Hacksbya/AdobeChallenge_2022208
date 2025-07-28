FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the processing script
COPY process_pdfs.py .

# Ensure input/output directories exist at runtime
RUN mkdir -p input output

# At runtime, scan /app/input for PDFs and write JSON to /app/output
CMD ["python", "process_pdfs.py"]
