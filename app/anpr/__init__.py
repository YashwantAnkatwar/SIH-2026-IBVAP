"""
app.anpr — Automatic Number Plate Recognition pipeline.

Real modular pipeline (no fabricated plate numbers, no hard-coded
strings): vehicle detection/tracking is reused from the existing
Detector/Tracker; this package adds plate localization, OCR, and
temporal aggregation on top.

Modules:
    plate_localizer — finds a candidate plate rectangle inside a
        vehicle's bounding box (cascade classifier + classical CV
        fallback). Offline, no model download required.
    ocr              — preprocesses a plate crop and runs Tesseract OCR,
        returning the raw recognized text plus Tesseract's own
        per-word confidence.
    aggregator       — per-vehicle-track temporal aggregation: many
        noisy OCR reads over time are reduced to one stable plate
        string, so we don't emit a new ANPR event every frame.
    pipeline         — wires the above into a single ANPRPipeline object
        the camera worker calls once per vehicle detection per frame.
"""
