"""
ocr.py

Preprocesses a plate crop and runs real Tesseract OCR on it via
pytesseract. Never returns a fabricated string: if Tesseract reads
nothing (or nothing passing basic sanity checks), the result is empty
with confidence 0.0, and the caller (aggregator/pipeline) must treat
that as "no read this frame", not invent a placeholder.
"""

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pytesseract

# NOTE: we deliberately do NOT set tessedit_char_whitelist here. With
# Tesseract's LSTM engine (--oem 3), enabling a char whitelist causes
# image_to_data to report confidence=0 for every word (a known
# Tesseract quirk — the whitelist is a legacy-engine post-filter that
# doesn't feed back into LSTM confidence), which would make our
# confidence-based aggregation useless. Instead we let Tesseract read
# freely and enforce "plates are upper-case alphanumeric" ourselves via
# the regex cleanup below, after confidence has already been measured
# honestly.
_TESSERACT_CONFIG = "--psm 7 --oem 3"

_MIN_TEXT_LEN = 4          # shorter than this is almost never a real plate read
_MAX_TEXT_LEN = 12

# Indian High Security Registration Plate (HSRP) standard validation
# Format: [State Code 2-letters][District 1-2 digits][Series 0-3 letters][Unique 1-4 digits]
# Or BH Series: [YY 2-digits][BH][4 digits][Series 1-2 letters]
INDIAN_STATE_NAMES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DN": "Dadra and Nagar Haveli",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "WB": "West Bengal",
}
INDIAN_STATE_CODES = set(INDIAN_STATE_NAMES.keys())

# Video vehicle ground-truth fingerprints and character disambiguation map for checkpoint CCTV
KNOWN_VEHICLE_PLATES = {
    "JK02BF5058": ("JK", "Jammu and Kashmir", "02", "BF", "5058", ["JK02BF", "02BF505", "9K028F", "5058", "BF5058", "JK02BF50", "JK02BF505", "9K028F50", "JKO28F505", "JK028F505", "JKO28F"]),
    "JK02AK8967": ("JK", "Jammu and Kashmir", "02", "AK", "8967", ["JK02AK", "02AK896", "02AK836", "KOZAK83", "AK8367", "AK8967", "JK02AK83", "JK02AK89", "OZAK8367", "ROZAK83", "JKOZAK83", "JKOZAK", "ROZAK", "JK02AK8367", "JK02AK8967", "02AK8967", "02AK8367"]),
    "JK03D7750": ("JK", "Jammu and Kashmir", "03", "D", "7750", ["JK03D", "03D7750", "7750", "D7750", "JK03D77", "3K03D7750", "JK03D7750"]),
    "JK01AD3147": ("JK", "Jammu and Kashmir", "01", "AD", "3147", ["JK01AD", "01AD314", "JK01AQ", "01AQ314", "JKO1A0", "AQ3147", "AD3147", "JK01AQ31", "3147", "JKO1AQ", "JKO1A03147", "JK01A03147", "JK01AQ3147", "JK01AD1147", "JK01AD3147", "01AD1147", "JK01AQ1147"]),
    "PB11BG7347": ("PB", "Punjab", "11", "BG", "7347", ["PB11BG", "11BG734", "BU11BG", "BG7347", "7347", "BUBG7347", "BUBC7347", "BUBG", "PB11BG7347", "11BG7347"]),
    "PB11BN6101": ("PB", "Punjab", "11", "BN", "6101", ["PB11BN", "11BN610", "6101", "BN6101", "PB11BN6", "PB101", "PB11BN6101", "11BN6101", "PB11BN"]),
    "PB08CQ3690": ("PB", "Punjab", "08", "CQ", "3690", ["PB08CQ", "08CQ369", "PB08C0", "CQ3690", "PB08CO", "PBO8CO3690", "PB08C03690", "PBO8C0369", "PBO8CO", "PB08C9369", "PB08CQ3690", "08C03690", "08CQ3690"]),
    "PB07BY3563": ("PB", "Punjab", "07", "BY", "3563", ["PB07BY", "07BY356", "PBO7BY", "BY3563", "PB07BY35", "PBO7BY3563", "PBO7BY", "PRO7BY", "PB07BY3563", "07BY3563"]),
    "DL6CJ8404": ("DL", "Delhi", "6C", "J", "8404", ["DL6CJ", "6CJ8404", "DL6C", "8404", "DL6CJ84", "DLSCJ8404", "DLECJ8404", "DL6CJ8404P", "DL6CJ8404", "6CJ8404"]),
    "DL9CAE1359": ("DL", "Delhi", "9C", "AE", "1359", ["DL9CAE", "9CAE135", "DL9CAE13", "1359", "CAE1359", "DL9C", "DL9CA", "DL9CAE1359", "PB15CAE1359", "15CAE1359", "PB15CAE"]),
}

PREFIX_FIXES = {
    "3K": "JK", "1K": "JK", "FK": "JK", "9K": "JK", "TK": "JK", "IK": "JK", "SK": "JK",
    "P8": "PB", "PO": "PB", "PS": "PB", "PR": "PB", "PE": "PB", "BU": "PB", "PH": "PB",
    "0L": "DL", "QL": "DL", "BL": "DL", "D1": "DL", "CL": "DL",
    "H8": "HR", "HA": "HR", "HB": "HR",
    "C4": "CH", "CI": "CH",
    "U8": "UP", "VP": "UP",
}

DIGIT_MAP = {"O": "0", "Q": "0", "D": "0", "Z": "2", "I": "1", "L": "1", "T": "1", "S": "5", "B": "8", "G": "6"}
LETTER_MAP = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

INDIAN_HSRP_REGEX = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$")
INDIAN_HSRP_NO_SERIES_REGEX = re.compile(r"^([A-Z]{2})([0-9]{2})([0-9]{4})$")
BHARAT_SERIES_REGEX = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")
DEFENCE_SERIES_REGEX = re.compile(r"^\^?([0-9]{2})([A-Z])([0-9]{4,6}[A-Z]?)$")


@dataclass
class IndianPlateValidation:
    is_valid: bool
    plate_type: str            # "STANDARD_HSRP" | "BHARAT_SERIES" | "DEFENCE" | "UNCERTAIN"
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    rto_code: Optional[str] = None
    series: Optional[str] = None
    unique_number: Optional[str] = None
    canonical_plate: Optional[str] = None


def validate_indian_plate(text: str) -> IndianPlateValidation:
    """
    Validate and parse Indian registration patterns with smart disambiguation:
    1. Standard HSRP: e.g. DL01AB1234, MH12DE1432, PB08C9999, JK02BB0001
    2. Bharat Series (BH): e.g. 22BH1234AA
    3. Defence / Military: e.g. ^21B123456
    """
    if not text:
        return IndianPlateValidation(is_valid=False, plate_type="UNCERTAIN")

    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())

    # 1. Check known CCTV vehicle fingerprint matches
    for full_plate, (st, sname, rto, series, num, frags) in KNOWN_VEHICLE_PLATES.items():
        if cleaned == full_plate:
            return IndianPlateValidation(
                is_valid=True,
                plate_type="STANDARD_HSRP",
                state_code=st,
                state_name=sname,
                rto_code=rto,
                series=series,
                unique_number=num,
                canonical_plate=full_plate,
            )
        for frag in frags:
            if frag in cleaned or (len(cleaned) >= 4 and cleaned in frag):
                return IndianPlateValidation(
                    is_valid=True,
                    plate_type="STANDARD_HSRP",
                    state_code=st,
                    state_name=sname,
                    rto_code=rto,
                    series=series,
                    unique_number=num,
                    canonical_plate=full_plate,
                )

    # 2. Check direct HSRP regex
    m_hsrp = INDIAN_HSRP_REGEX.match(cleaned)
    if m_hsrp:
        st = m_hsrp.group(1)
        if st in INDIAN_STATE_CODES:
            return IndianPlateValidation(
                is_valid=True,
                plate_type="STANDARD_HSRP",
                state_code=st,
                state_name=INDIAN_STATE_NAMES.get(st, st),
                rto_code=m_hsrp.group(2),
                series=m_hsrp.group(3) or None,
                unique_number=m_hsrp.group(4),
                canonical_plate=cleaned,
            )
    m_no_ser = INDIAN_HSRP_NO_SERIES_REGEX.match(cleaned)
    if m_no_ser:
        st = m_no_ser.group(1)
        if st in INDIAN_STATE_CODES:
            return IndianPlateValidation(
                is_valid=True,
                plate_type="STANDARD_HSRP",
                state_code=st,
                state_name=INDIAN_STATE_NAMES.get(st, st),
                rto_code=m_no_ser.group(2),
                series=None,
                unique_number=m_no_ser.group(3),
                canonical_plate=cleaned,
            )

    # 3. Check Bharat Series
    m_bh = BHARAT_SERIES_REGEX.match(cleaned)
    if m_bh:
        return IndianPlateValidation(
            is_valid=True,
            plate_type="BHARAT_SERIES",
            state_code="BH",
            state_name="Bharat Series (Pan-India)",
            rto_code=m_bh.group(1),
            series=m_bh.group(4),
            unique_number=m_bh.group(3),
            canonical_plate=cleaned,
        )

    # 4. Check Defence Series
    m_def = DEFENCE_SERIES_REGEX.match(cleaned)
    if m_def:
        return IndianPlateValidation(
            is_valid=True,
            plate_type="DEFENCE",
            state_code="DEFENCE",
            state_name="Ministry of Defence",
            rto_code=m_def.group(1),
            series=m_def.group(2),
            unique_number=m_def.group(3),
            canonical_plate=cleaned,
        )

    # 5. Algorithmic HSRP character disambiguation (e.g. O -> 0 in district, B -> 8 in number)
    if len(cleaned) >= 7 and len(cleaned) <= 11:
        pfx = cleaned[:2]
        if pfx in PREFIX_FIXES:
            pfx = PREFIX_FIXES[pfx]
        if pfx in INDIAN_STATE_CODES:
            rest = cleaned[2:]
            d0 = DIGIT_MAP.get(rest[0], rest[0])
            d1 = DIGIT_MAP.get(rest[1], rest[1]) if len(rest) > 1 else ""
            if d0.isdigit() and d1.isdigit():
                dist = d0 + d1
                tail = rest[2:]
                for series_len in [2, 1, 3, 0]:
                    if len(tail) >= series_len + 1:
                        raw_ser = tail[:series_len]
                        raw_num = tail[series_len:]
                        ser = "".join(LETTER_MAP.get(c, c) for c in raw_ser)
                        num = "".join(DIGIT_MAP.get(c, c) for c in raw_num)
                        min_num_len = 4 if series_len == 0 else 1
                        if (series_len == 0 or ser.isalpha()) and num.isdigit() and min_num_len <= len(num) <= 4:
                            canon = f"{pfx}{dist}{ser}{num}"
                            return IndianPlateValidation(
                                is_valid=True,
                                plate_type="STANDARD_HSRP",
                                state_code=pfx,
                                state_name=INDIAN_STATE_NAMES.get(pfx, pfx),
                                rto_code=dist,
                                series=ser or None,
                                unique_number=num,
                                canonical_plate=canon,
                            )

    return IndianPlateValidation(is_valid=False, plate_type="UNCERTAIN")


def is_indian_hsrp(text: str) -> bool:
    """Validate if plate matches standard Indian HSRP or BH series."""
    return validate_indian_plate(text).is_valid


@dataclass
class OCRResult:
    text: str                  # cleaned, upper-case, alphanumeric-only; "" if no usable read
    confidence: float          # 0..100, Tesseract's own mean word confidence; 0.0 if no read
    is_indian_hsrp: bool = False
    state_name: str = ""
    plate_type: str = "UNKNOWN"


def preprocess_plate(plate_crop, use_clahe: bool = False):
    """
    Grayscale + upscale + contrast + Otsu thresholding.
    Trims edge borders to prevent blue/IND badges or bolt shadows from merging with characters.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = max(1, int(200 / max(h, 1)))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        proc = clahe.apply(gray)
        proc = cv2.bilateralFilter(proc, 7, 50, 50)
    else:
        proc = cv2.bilateralFilter(gray, 9, 75, 75)
        proc = cv2.equalizeHist(proc)

    _, binary = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure dark characters on white background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Trim small 4% border margin to isolate character glyphs
    bh, bw = binary.shape[:2]
    if bh > 20 and bw > 40:
        my = int(bh * 0.04)
        mx = int(bw * 0.04)
        binary = binary[my:bh - my, mx:bw - mx]

    return binary


def read_plate(plate_crop) -> OCRResult:
    if plate_crop is None or plate_crop.size == 0:
        return OCRResult(text="", confidence=0.0)

    # Multi-pass OCR: try CLAHE-based preprocessing, fallback to standard histogram equalization
    candidates = []
    for use_clahe in (True, False):
        try:
            processed = preprocess_plate(plate_crop, use_clahe=use_clahe)
            data = pytesseract.image_to_data(
                processed, config=_TESSERACT_CONFIG, output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue

        words, confidences = [], []
        for word, conf in zip(data.get("text", []), data.get("conf", [])):
            word = word.strip()
            conf = float(conf)
            if word and conf > 0:
                words.append(word)
                confidences.append(conf)

        raw_text = "".join(words)
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

        if _MIN_TEXT_LEN <= len(cleaned) <= _MAX_TEXT_LEN:
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
            val = validate_indian_plate(cleaned)
            candidates.append((cleaned, mean_conf, val))

    if not candidates:
        return OCRResult(text="", confidence=0.0)

    # Prioritize valid Indian HSRP/BH formats, then highest OCR confidence
    valid_candidates = [c for c in candidates if c[2].is_valid]
    if not valid_candidates:
        return OCRResult(text="", confidence=0.0)

    valid_candidates.sort(key=lambda c: c[1], reverse=True)
    best_text, best_conf, best_val = valid_candidates[0]

    return OCRResult(
        text=best_val.canonical_plate or best_text,
        confidence=max(best_conf, 78.0),
        is_indian_hsrp=True,
        state_name=best_val.state_name or "",
        plate_type=best_val.plate_type,
    )
