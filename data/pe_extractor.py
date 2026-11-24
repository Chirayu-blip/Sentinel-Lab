"""Extract the compact static feature set used by the detector."""

import re
from collections.abc import Mapping

import numpy as np

STATIC_FEATURES = [
    "file_size",
    "entropy",
    "num_imports",
    "num_exports",
    "num_sections",
    "has_debug",
    "has_tls",
    "has_resources",
    "virtual_size",
    "raw_size",
    "section_entropy_mean",
    "suspicious_imports",
    "packer_detected",
    "timestamp_anomaly",
]

SUSPICIOUS_IMPORT_NAMES = {
    "virtualalloc",
    "virtualallocex",
    "writeprocessmemory",
    "createremotethread",
    "ntunmapviewofsection",
    "setthreadcontext",
    "internetopenurl",
    "urldownloadtofilea",
    "urldownloadtofilew",
    "winexec",
    "shellexecutea",
    "shellexecutew",
}
PACKER_SECTION = re.compile(r"(upx|aspack|mpress|themida|vmp|packed)", re.I)


def byte_entropy(content: bytes) -> float:
    if not content:
        return 0.0
    counts = np.bincount(np.frombuffer(content, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(content)
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_pe_features(content: bytes) -> dict[str, float]:
    try:
        import pefile
    except ImportError as exc:
        raise RuntimeError("pefile is required to inspect PE files") from exc

    try:
        pe = pefile.PE(data=content, fast_load=False)
    except (pefile.PEFormatError, ValueError) as exc:
        raise ValueError(f"Unable to parse this file as a Windows PE: {exc}") from exc

    sections = pe.sections
    imports = [
        entry.name.decode("ascii", errors="ignore").lower()
        for descriptor in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        for entry in descriptor.imports
        if entry.name
    ]
    exports = getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", [])
    section_entropies = [float(section.get_entropy()) for section in sections]
    section_names = [
        section.Name.rstrip(b"\0").decode("ascii", errors="ignore")
        for section in sections
    ]
    timestamp = int(getattr(pe.FILE_HEADER, "TimeDateStamp", 0))
    # Dates outside this broad range are an anomaly indicator, not proof.
    timestamp_anomaly = float(timestamp < 631152000 or timestamp > 2145916800)
    return {
        "file_size": float(len(content)),
        "entropy": byte_entropy(content),
        "num_imports": float(len(imports)),
        "num_exports": float(len(exports)),
        "num_sections": float(len(sections)),
        "has_debug": float(bool(getattr(pe, "DIRECTORY_ENTRY_DEBUG", []))),
        "has_tls": float(bool(getattr(pe, "DIRECTORY_ENTRY_TLS", None))),
        "has_resources": float(bool(getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None))),
        "virtual_size": float(sum(int(section.Misc_VirtualSize) for section in sections)),
        "raw_size": float(sum(int(section.SizeOfRawData) for section in sections)),
        "section_entropy_mean": float(np.mean(section_entropies)) if sections else 0.0,
        "suspicious_imports": float(
            sum(name in SUSPICIOUS_IMPORT_NAMES for name in imports)
        ),
        "packer_detected": float(
            any(PACKER_SECTION.search(name) for name in section_names)
            or any(value >= 7.2 for value in section_entropies)
        ),
        "timestamp_anomaly": timestamp_anomaly,
    }


def features_from_ember(record: Mapping) -> dict[str, float]:
    """Map a raw EMBER JSON record into the same compact feature schema."""
    general = record.get("general") or {}
    imports_by_dll = record.get("imports") or {}
    imported_names = [
        str(function).split(":", 1)[0].lower()
        for functions in imports_by_dll.values()
        for function in (functions if isinstance(functions, list) else [])
    ] if isinstance(imports_by_dll, Mapping) else []
    exports = record.get("exports") or []
    section_data = record.get("sections") or {}
    section_list = section_data.get("sections", []) if isinstance(section_data, Mapping) else []
    section_list = section_list if isinstance(section_list, list) else []
    entropies = [float(section.get("entropy") or 0) for section in section_list]
    names = [str(section.get("name") or "") for section in section_list]
    directories = record.get("datadirectories") or record.get("data_directories") or []
    directory_names = {
        str(item.get("name", "")).lower()
        for item in directories
        if isinstance(item, Mapping)
    } if isinstance(directories, list) else set()
    header = record.get("header") or {}
    timestamp = int(header.get("timestamp") or 0) if isinstance(header, Mapping) else 0
    histogram = record.get("histogram")
    if isinstance(histogram, list) and histogram:
        histogram_counts = np.asarray(histogram, dtype=float)
        total = histogram_counts.sum()
        probabilities = histogram_counts[histogram_counts > 0] / total if total else []
        entropy = float(-(probabilities * np.log2(probabilities)).sum()) if len(probabilities) else 0.0
    else:
        entropy = float(general.get("entropy") or 0)

    return {
        "file_size": float(general.get("size") or 0),
        "entropy": entropy,
        "num_imports": float(sum(len(fs) for fs in imports_by_dll.values() if isinstance(fs, list)))
        if isinstance(imports_by_dll, Mapping) else 0.0,
        "num_exports": float(len(exports)) if isinstance(exports, list) else 0.0,
        "num_sections": float(len(section_list)),
        "has_debug": float(any("debug" in name for name in directory_names)),
        "has_tls": float(any("tls" in name for name in directory_names)),
        "has_resources": float(any("resource" in name for name in directory_names)),
        "virtual_size": float(sum(float(section.get("vsize") or 0) for section in section_list)),
        "raw_size": float(sum(float(section.get("size") or 0) for section in section_list)),
        "section_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "suspicious_imports": float(sum(name in SUSPICIOUS_IMPORT_NAMES for name in imported_names)),
        "packer_detected": float(
            any(PACKER_SECTION.search(name) for name in names)
            or any(value >= 7.2 for value in entropies)
        ),
        "timestamp_anomaly": float(timestamp < 631152000 or timestamp > 2145916800),
    }
