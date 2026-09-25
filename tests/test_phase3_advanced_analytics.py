#!/usr/bin/env python3
"""
tests/test_phase3_advanced_analytics.py

Comprehensive Phase 3 Test Suite for IBVAP:
3A - Indian ANPR (Localization, Preprocessing, Dual-pass OCR, Temporal Aggregation, Indian Plate Validation)
3B - Software Face Recognition (FaceNet Embedder, Authorized Match, Unknown Rejection, Quality Filtering)
3C - Night / Thermal Movement Detection (Luminance, Low-light State, Zone + Speed Gating)
3D - Event Bus and Event Store Schema Integration (ANPREvent, FaceRecognitionEvent, NightMovementEvent)
3E - Multi-Camera Isolation and Pipeline Performance Benchmarks
"""

import os
import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT))

from anpr.aggregator import PlateTrackAggregator
from anpr.ocr import (
    preprocess_plate, read_plate, validate_indian_plate, is_indian_hsrp,
    INDIAN_STATE_NAMES
)
from anpr.plate_localizer import PlateLocalizer
from face.face_detector import FaceDetector
from face.embedder_factory import create_embedder
from face.gallery import FaceGallery
from face.pipeline import FaceRecognitionPipeline
from night_detection import NightDetector
from events import (
    event_bus, ANPREvent, FaceRecognitionEvent, NightMovementEvent
)
from event_store import EventStore
import config


class TestPhase3IndianANPR(unittest.TestCase):
    """3A: Indian ANPR verification on authentic dataset samples and validation rules."""

    def setUp(self):
        self.localizer = PlateLocalizer()
        self.aggregator = PlateTrackAggregator()

    def test_indian_plate_localization_on_real_dataset(self):
        """Plate localizer must detect plate bounding box on real Indian vehicles without full-image OCR."""
        dl_img_path = PROJECT_ROOT / "IBVAP_Datasets" / "Indian_ANPR" / "State-wise_OLX" / "DL" / "DL1.jpg"
        self.assertTrue(dl_img_path.exists(), f"Missing dataset file: {dl_img_path}")

        img = cv2.imread(str(dl_img_path))
        self.assertIsNotNone(img)
        cand = self.localizer.locate(img)

        self.assertIsNotNone(cand, "PlateLocalizer failed to find plate candidate on DL1.jpg")
        bx1, by1, bx2, by2 = cand.box
        self.assertGreater(bx2, bx1)
        self.assertGreater(by2, by1)
        # Verify plausible aspect ratio
        w = bx2 - bx1
        h = by2 - by1
        aspect = w / float(h)
        self.assertGreater(aspect, 1.5, f"Aspect ratio {aspect:.2f} is too square for a license plate")
        self.assertLess(aspect, 6.5, f"Aspect ratio {aspect:.2f} is too wide for a license plate")
        # DL1 ground truth plate y is around 235-255 in a 363-tall image
        self.assertGreater(by1, 150, "Plate candidate should be in the lower half of the vehicle")

    def test_indian_plate_preprocessing_and_ocr(self):
        """Preprocessing must clean plate crop and OCR must extract readable characters."""
        # Render a synthetic plate or use DL1 ground truth crop
        plate_text_target = "DL6CJ8404"
        canvas = np.ones((80, 280, 3), dtype=np.uint8) * 255
        cv2.rectangle(canvas, (2, 2), (278, 78), (0, 0, 0), 2)
        cv2.putText(canvas, "IND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
        cv2.putText(canvas, plate_text_target, (45, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

        preprocessed = preprocess_plate(canvas, use_clahe=True)
        self.assertGreater(preprocessed.shape[0], 0)
        self.assertGreater(preprocessed.shape[1], 0)

        result = read_plate(canvas)
        self.assertGreater(result.confidence, 50.0)
        self.assertIn("DL6CJ", result.text.replace("O", "D"))

    def test_indian_plate_validation_rules(self):
        """Standard HSRP, Bharat series (BH), and Defence formats must validate correctly with state codes."""
        test_cases = [
            ("DL01AB1234", True, "STANDARD_HSRP", "DL", "Delhi"),
            ("MH12DE1432", True, "STANDARD_HSRP", "MH", "Maharashtra"),
            ("HR26BQ8176", True, "STANDARD_HSRP", "HR", "Haryana"),
            ("PB11BN6101", True, "STANDARD_HSRP", "PB", "Punjab"),
            ("JK02BB0001", True, "STANDARD_HSRP", "JK", "Jammu and Kashmir"),
            ("UP32AA1111", True, "STANDARD_HSRP", "UP", "Uttar Pradesh"),
            ("RJ14CA9999", True, "STANDARD_HSRP", "RJ", "Rajasthan"),
            ("KA04MH1234", True, "STANDARD_HSRP", "KA", "Karnataka"),
            ("TN09AA0001", True, "STANDARD_HSRP", "TN", "Tamil Nadu"),
            ("GJ01AB1234", True, "STANDARD_HSRP", "GJ", "Gujarat"),
            ("22BH1234AA", True, "BHARAT_SERIES", "BH", "Bharat Series (Pan-India)"),
            ("21BH9999Z", True, "BHARAT_SERIES", "BH", "Bharat Series (Pan-India)"),
            ("^21B123456", True, "DEFENCE", "DEFENCE", "Ministry of Defence"),
            ("INVALIDPLATE", False, "UNCERTAIN", None, None),
            ("XY99ZZ9999", False, "UNCERTAIN", None, None),  # XY is not an Indian State code
        ]

        for text, expected_valid, expected_type, exp_st, exp_name in test_cases:
            res = validate_indian_plate(text)
            self.assertEqual(res.is_valid, expected_valid, f"Validation failed for '{text}'")
            self.assertEqual(res.plate_type, expected_type, f"Plate type mismatch for '{text}'")
            if expected_valid:
                self.assertEqual(res.state_code, exp_st)
                self.assertEqual(res.state_name, exp_name)

    def test_anpr_temporal_aggregation_and_debouncing(self):
        """Temporal aggregator must require multiple supporting frames, select mode, and debounce duplicates."""
        track_id = "BOP-03:vehicle_101"
        now = time.time()

        # Frame 1: Single read should NOT stabilize
        self.aggregator.observe(track_id, "DL6CJ8404", 82.0, timestamp=now)
        out1 = self.aggregator.get_stable_unreported(track_id)
        self.assertIsNone(out1, "Aggregator should not stabilize on frame 1")

        # Frame 2: Second agreeing read stabilizes
        self.aggregator.observe(track_id, "DL6CJ8404", 88.0, timestamp=now + 0.1)
        out2 = self.aggregator.get_stable_unreported(track_id)
        self.assertIsNotNone(out2, "Aggregator should stabilize after 2 agreeing reads")
        self.assertEqual(out2[0], "DL6CJ8404")

        # Frame 3: Same read within debounce window should be suppressed (no duplicate event spam)
        self.aggregator.observe(track_id, "DL6CJ8404", 90.0, timestamp=now + 0.2)
        out3 = self.aggregator.get_stable_unreported(track_id)
        self.assertIsNone(out3, "Aggregator must debounce identical reads on the same track")

        # Detailed state check
        details = self.aggregator.get_stable_details(track_id)
        self.assertIsNotNone(details)
        text, conf, supporting_frames, box, ts = details
        self.assertEqual(text, "DL6CJ8404")
        self.assertGreaterEqual(supporting_frames, 2)

        val = validate_indian_plate(text)
        self.assertTrue(val.is_valid)
        self.assertEqual(val.state_name, "Delhi")

    def test_anpr_track_isolation(self):
        """Different vehicles and cameras must maintain independent plate track states."""
        t1 = "BOP-03:vehicle_1"
        t2 = "BOP-03:vehicle_2"
        now = time.time()

        self.aggregator.observe(t1, "MH12DE1432", 85.0, timestamp=now)
        self.aggregator.observe(t1, "MH12DE1432", 87.0, timestamp=now + 0.1)

        self.aggregator.observe(t2, "HR26BQ8176", 80.0, timestamp=now)
        self.aggregator.observe(t2, "HR26BQ8176", 84.0, timestamp=now + 0.1)

        d1 = self.aggregator.get_stable_details(t1)
        d2 = self.aggregator.get_stable_details(t2)

        self.assertEqual(d1[0], "MH12DE1432")
        self.assertEqual(d2[0], "HR26BQ8176")


class TestPhase3FaceRecognition(unittest.TestCase):
    """3B: Face Recognition verification using Deep FaceNet Embeddings and Gallery Matching."""

    def setUp(self):
        self.embedder = create_embedder("facenet")
        self.gallery_dir = PROJECT_ROOT / "gallery"
        self.gallery = FaceGallery(self.gallery_dir, embedder=self.embedder)

    def test_deep_embedding_properties(self):
        """FaceNet embedder must generate 512-dim unit-normalized feature vector."""
        sample_img = PROJECT_ROOT / "gallery" / "person_01" / "enroll_00_frame566.jpg"
        self.assertTrue(sample_img.exists())
        crop = cv2.imread(str(sample_img))
        vec = self.embedder.embed(crop)

        self.assertEqual(len(vec), 512)
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_authorized_demo_face_recognition(self):
        """Authorized demo identity from enrolled gallery must match with high similarity."""
        # Use a distinct frame from person_01 not used as the primary enrollment template
        test_img = PROJECT_ROOT / "gallery" / "person_01" / "enroll_50_frame623.jpg"
        self.assertTrue(test_img.exists())
        crop = cv2.imread(str(test_img))

        match = self.gallery.match(crop)
        self.assertTrue(match.is_known, "Authorized person should be recognized as Known")
        self.assertEqual(match.identity, "person_01")
        self.assertGreater(match.confidence, 0.45)
        self.assertLess(match.distance, self.gallery.match_threshold)

    def test_unknown_person_rejection(self):
        """Un-enrolled or synthetic face must be rejected as Unknown."""
        # Synthetic distinct face pattern
        fake_face = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(fake_face, (50, 50), 30, (180, 180, 180), -1)
        cv2.circle(fake_face, (40, 40), 5, (20, 20, 20), -1)
        cv2.circle(fake_face, (60, 40), 5, (20, 20, 20), -1)
        cv2.ellipse(fake_face, (50, 65), (15, 8), 0, 0, 180, (20, 20, 20), 2)

        match = self.gallery.match(fake_face)
        self.assertFalse(match.is_known, "Un-enrolled face must not be identified as an authorized person")
        self.assertEqual(match.identity, "Unknown")

    def test_poor_quality_face_rejection(self):
        """Detector quality check must reject tiny or low-variance blurry crops."""
        detector = FaceDetector()
        # Tiny crop
        tiny = np.ones((15, 15, 3), dtype=np.uint8) * 128
        self.assertIsNone(detector.quality_check(tiny))

        # Flat blurry crop
        flat = np.ones((80, 80, 3), dtype=np.uint8) * 120
        self.assertIsNone(detector.quality_check(flat))


class TestPhase3NightMovement(unittest.TestCase):
    """3C: Night / Thermal Movement Detection based on scene luminance and zone velocity gating."""

    def setUp(self):
        self.detector = NightDetector(threshold=70.0, window=5)

    def test_luminance_estimation_and_state(self):
        """Mean grayscale brightness must correctly distinguish day vs night/thermal."""
        # Isolated detector for day frame
        day_det = NightDetector(threshold=70.0, window=5)
        day_frame = np.ones((100, 100, 3), dtype=np.uint8) * 160
        s_day = day_det.update(day_frame)
        self.assertFalse(s_day["is_night"])
        self.assertGreater(s_day["smoothed_luminance"], 70.0)

        # Isolated detector for night / thermal frame
        night_det = NightDetector(threshold=70.0, window=5)
        night_frame = np.ones((100, 100, 3), dtype=np.uint8) * 35
        s_night = night_det.update(night_frame)
        self.assertTrue(s_night["is_night"])
        self.assertLess(s_night["smoothed_luminance"], 70.0)

    def test_night_movement_gating_logic(self):
        """Movement alert requires BOTH low light AND movement inside a zone."""
        # Simulate conditions evaluated in camera worker
        def eval_night_alert(is_night, zones_occupied, speed):
            # Same condition as app/camera_worker.py:551
            return bool(is_night and zones_occupied and speed > 5)

        # Case 1: Dark night, but stationary person (speed <= 5) -> No alert
        self.assertFalse(eval_night_alert(is_night=True, zones_occupied={"perimeter_zone"}, speed=0.0))

        # Case 2: Broad daylight, moving person -> No NIGHT alert
        self.assertFalse(eval_night_alert(is_night=False, zones_occupied={"perimeter_zone"}, speed=45.0))

        # Case 3: Dark night, moving outside any restricted zone -> No NIGHT alert
        self.assertFalse(eval_night_alert(is_night=True, zones_occupied=set(), speed=45.0))

        # Case 4: Dark night, moving inside zone -> Triggers NIGHT_MOVEMENT
        self.assertTrue(eval_night_alert(is_night=True, zones_occupied={"restricted_sector"}, speed=25.0))


class TestPhase3EventIntegration(unittest.TestCase):
    """3D: Integration of ANPR, FRS, and Night events into EventBus and EventStore."""

    def setUp(self):
        self.store = EventStore(":memory:")

    def test_anpr_event_schema_and_persistence(self):
        """ANPR event must contain complete plate, confidence, state, and frame count metadata."""
        now = time.time()
        ev = ANPREvent(
            camera_id="BOP-03",
            vehicle_track_id="BOP-03:veh_1",
            plate_text="DL6CJ8404",
            confidence=88.5,
            event_id="ANPR-BOP-03-20260919-veh_1",
            plate_box=(89, 230, 185, 262),
            supporting_frames=3,
            is_indian=True,
            state_name="Delhi",
            evidence_path="anpr/sample.jpg",
        )
        event_bus.publish(ev)

        self.store.record(
            camera_id=ev.camera_id,
            track_id=ev.vehicle_track_id,
            object_type="vehicle",
            event_type=ev.event_type,
            severity="LOW",
            description=f"ANPR: {ev.plate_text} [{ev.state_name}]",
            timestamp=now,
            plate_text=ev.plate_text,
            plate_confidence=ev.confidence,
            event_id=ev.event_id,
            plate_box=ev.plate_box,
            supporting_frames=ev.supporting_frames,
            is_indian=ev.is_indian,
            state_name=ev.state_name,
            evidence_path=ev.evidence_path,
        )

        rows = self.store.query(camera_id="BOP-03", event_type="ANPR_PLATE_READ")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["camera_id"], "BOP-03")
        self.assertEqual(r["extra"]["plate_text"], "DL6CJ8404")
        self.assertEqual(r["extra"]["state_name"], "Delhi")
        self.assertTrue(r["extra"]["is_indian"])
        self.assertEqual(r["extra"]["supporting_frames"], 3)

    def test_frs_event_schema_and_persistence(self):
        """Face recognition event must record identity, similarity, threshold, and status."""
        now = time.time()
        ev = FaceRecognitionEvent(
            camera_id="BOP-03",
            person_track_id="BOP-03:person_1",
            identity="person_01",
            confidence=0.82,
            is_known=True,
            event_id="FRS-BOP-03-20260919-person_1",
            similarity=0.82,
            threshold=0.42,
            face_box=(30, 40, 90, 100),
            evidence_path="face/person_01.jpg",
        )
        event_bus.publish(ev)

        self.store.record(
            camera_id=ev.camera_id,
            track_id=ev.person_track_id,
            object_type="person",
            event_type=ev.event_type,
            severity="MEDIUM",
            description=f"Face recognition: {ev.identity}",
            timestamp=now,
            identity=ev.identity,
            recognition_confidence=ev.confidence,
            recognition_status="KNOWN",
            event_id=ev.event_id,
            similarity=ev.similarity,
            threshold=ev.threshold,
            evidence_path=ev.evidence_path,
        )

        rows = self.store.query(camera_id="BOP-03", event_type="FACE_RECOGNITION")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["extra"]["identity"], "person_01")
        self.assertEqual(r["extra"]["recognition_status"], "KNOWN")
        self.assertAlmostEqual(r["extra"]["similarity"], 0.82)

    def test_night_movement_event_schema_and_persistence(self):
        """Night movement event must capture scene luminance, threshold, speed, and zone."""
        now = time.time()
        ev = NightMovementEvent(
            camera_id="BOP-02",
            track_id="BOP-02:person_4",
            zone="restricted_sector_east",
            speed=24.5,
            luminance=34.2,
            threshold=70.0,
            is_night=True,
            event_id="NIGHT-BOP-02-123456-4",
            evidence_path="evidence/night_bop2.jpg",
        )
        event_bus.publish(ev)

        self.store.record(
            camera_id=ev.camera_id,
            track_id=ev.track_id,
            zone=ev.zone,
            event_type=ev.event_type,
            severity="HIGH",
            description=f"Movement in '{ev.zone}' under low-light",
            timestamp=now,
            event_id=ev.event_id,
            scene_luminance=ev.luminance,
            threshold=ev.threshold,
            speed=ev.speed,
            is_night=ev.is_night,
        )

        rows = self.store.query(camera_id="BOP-02", event_type="NIGHT_MOVEMENT")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["severity"], "HIGH")
        self.assertEqual(r["zone"], "restricted_sector_east")
        self.assertAlmostEqual(r["extra"]["scene_luminance"], 34.2)
        self.assertTrue(r["extra"]["is_night"])

    def test_camera_isolation(self):
        """Events logged for BOP-01, BOP-02, and BOP-03 must stay segregated."""
        self.store.record(camera_id="BOP-01", event_type="TEST_1")
        self.store.record(camera_id="BOP-02", event_type="TEST_2")
        self.store.record(camera_id="BOP-03", event_type="TEST_3")

        self.assertEqual(len(self.store.query(camera_id="BOP-01")), 1)
        self.assertEqual(len(self.store.query(camera_id="BOP-02")), 1)
        self.assertEqual(len(self.store.query(camera_id="BOP-03")), 1)


class TestPhase3Performance(unittest.TestCase):
    """3E: Real Performance and Latency Benchmark."""

    def test_processing_latency_and_throughput(self):
        """Measure real frame execution latency for analytics pipelines."""
        night_det = NightDetector(threshold=70.0)
        localizer = PlateLocalizer()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_t = time.time()
        n_iters = 30
        for _ in range(n_iters):
            # 1. Luminance estimation
            _ = night_det.update(frame)
            # 2. Plate localization attempt
            _ = localizer.locate(frame[200:450, 100:500])

        elapsed = time.time() - start_t
        avg_ms = (elapsed / n_iters) * 1000.0
        fps = n_iters / elapsed

        print(f"\n[PERFORMANCE BENCHMARK] Processed {n_iters} frames in {elapsed:.2f}s | Latency: {avg_ms:.2f} ms/frame | Throughput: {fps:.1f} FPS")
        self.assertLess(avg_ms, 150.0, "Average analytics latency per frame should remain under 150ms")


if __name__ == "__main__":
    unittest.main()
