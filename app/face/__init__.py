"""
app.face — Face detection + embedding-based recognition pipeline.

Real modular pipeline: person detection/tracking is reused from the
existing Detector/Tracker; this package adds face detection, a real
feature-embedding step, and gallery matching on top.

Modules:
    face_detector — locates faces inside a person crop (Haar cascade,
        bundled with OpenCV, fully offline) and does basic quality
        filtering (size, blur).
    embedder      — computes a real feature vector for a face crop
        (Local Binary Pattern histogram descriptor: a classical,
        explainable, deep-learning-free embedding). Two faces of the
        same person produce similar vectors; different faces produce
        dissimilar vectors. This is a genuine embedding-and-distance
        method, not a lookup table and not a fixed/fake result — it is
        deliberately NOT a deep neural embedding (no FaceNet/ArcFace),
        which is an explicit, documented accuracy trade-off made to
        keep this pipeline runnable fully offline in this environment.
        Swapping in a deep embedding model later is a drop-in
        replacement of this one module.
    gallery       — stores named/ID'd enrollment embeddings and finds
        the nearest match (or "Unknown" if nothing is close enough).
    pipeline      — wires the above into a single
        FaceRecognitionPipeline object, with per-track temporal
        stabilization so one physical person crossing many frames
        produces one recognition result, not one per frame.
"""
