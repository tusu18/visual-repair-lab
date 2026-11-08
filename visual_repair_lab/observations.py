"""Optional pixel-derived text baseline. Never reads evaluator truth or state."""
import io
import shutil
import subprocess
from PIL import Image


class ObservationAdapterError(RuntimeError):
    pass


def pixel_text(image, timeout=10):
    binary=shutil.which('tesseract')
    if binary is None:
        raise ObservationAdapterError('ocr_dependency_missing')
    data=io.BytesIO();Image.fromarray(image).save(data,format='PNG')
    try:
        process=subprocess.run([binary,'stdin','stdout','--psm','11'],input=data.getvalue(),
            capture_output=True,timeout=timeout,check=False)
    except (OSError,subprocess.TimeoutExpired):
        raise ObservationAdapterError('ocr_worker_failed') from None
    if process.returncode:
        raise ObservationAdapterError('ocr_worker_failed')
    text=process.stdout.decode('utf-8',errors='replace')
    if len(text)>100000:
        raise ObservationAdapterError('ocr_output_limit')
    return text
