"""Normalize API Monitor CSV exports into the dynamic feature schema."""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

DYNAMIC_FEATURES = [
    "VirtualAlloc",
    "WriteProcessMemory",
    "CreateRemoteThread",
    "OpenProcess",
    "InternetOpenUrl",
    "RegSetValue",
    "CreateService",
    "CryptEncrypt",
    "ShellExecute",
    "GetAsyncKeyState",
    "CreateFile",
    "DeleteFile",
    "unique_api_count",
    "total_api_calls",
    "network_activity",
    "registry_writes",
    "file_deletions",
]

API_ALIASES = {
    "VirtualAlloc": ("VirtualAlloc", "VirtualAllocEx", "NtAllocateVirtualMemory"),
    "WriteProcessMemory": ("WriteProcessMemory", "NtWriteVirtualMemory"),
    "CreateRemoteThread": ("CreateRemoteThread", "CreateRemoteThreadEx", "RtlCreateUserThread"),
    "OpenProcess": ("OpenProcess", "NtOpenProcess"),
    "InternetOpenUrl": ("InternetOpenUrl", "InternetConnect", "HttpOpenRequest", "URLDownloadToFile"),
    "RegSetValue": ("RegSetValue", "RegSetValueEx", "RegCreateKey", "RegCreateKeyEx"),
    "CreateService": ("CreateService", "OpenSCManager"),
    "CryptEncrypt": ("CryptEncrypt", "BCryptEncrypt", "CryptAcquireContext"),
    "ShellExecute": ("ShellExecute", "ShellExecuteEx", "WinExec", "CreateProcess"),
    "GetAsyncKeyState": ("GetAsyncKeyState", "SetWindowsHookEx"),
    "CreateFile": ("CreateFile", "NtCreateFile"),
    "DeleteFile": ("DeleteFile", "NtDeleteFile"),
}
API_COLUMN_CANDIDATES = ("function", "api", "api_name", "function_name", "name", "call")


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def parse_apimon_csv(path_or_file) -> dict[str, float]:
    frame = pd.read_csv(path_or_file)
    if frame.empty:
        raise ValueError("The API Monitor CSV contains no API call rows")
    columns = {_normalize(column): column for column in frame.columns}
    api_column = next(
        (columns[_normalize(candidate)] for candidate in API_COLUMN_CANDIDATES
         if _normalize(candidate) in columns),
        None,
    )
    if api_column is None:
        raise ValueError(
            "Could not identify the API-name column. Expected one of: "
            + ", ".join(API_COLUMN_CANDIDATES)
        )
    calls = frame[api_column].dropna().astype(str).tolist()
    normalized_calls = [_normalize(call) for call in calls]
    counts = Counter(normalized_calls)
    output = {feature: 0.0 for feature in DYNAMIC_FEATURES}
    for feature, aliases in API_ALIASES.items():
        alias_set = {_normalize(alias) for alias in aliases}
        output[feature] = float(
            sum(count for call, count in counts.items()
                if any(
                    call.endswith(alias)
                    or call.endswith(alias + "a")
                    or call.endswith(alias + "w")
                    for alias in alias_set
                ))
        )
    output["unique_api_count"] = float(len(counts))
    output["total_api_calls"] = float(len(calls))
    output["network_activity"] = output["InternetOpenUrl"]
    output["registry_writes"] = output["RegSetValue"]
    output["file_deletions"] = output["DeleteFile"]
    return output


def load_labeled_captures(manifest_path: str) -> pd.DataFrame:
    """Build one labeled feature row per API Monitor CSV in a capture manifest."""
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"capture_path", "label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            "Capture manifest is missing columns: " + ", ".join(sorted(missing))
        )
    rows = []
    for row in manifest.itertuples(index=False):
        label = int(row.label)
        if label not in (0, 1):
            raise ValueError("Capture manifest labels must be 0 (benign) or 1 (malware)")
        capture_path = Path(row.capture_path)
        if not capture_path.is_absolute():
            capture_path = manifest_path.parent / capture_path
        features = parse_apimon_csv(capture_path)
        features["label"] = label
        rows.append(features)
    if not rows:
        raise ValueError("Capture manifest contains no labeled captures")
    return pd.DataFrame(rows, columns=DYNAMIC_FEATURES + ["label"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with capture_path,label columns")
    parser.add_argument("--output", default="data/dynamic_labeled.csv")
    args = parser.parse_args()
    frame = load_labeled_captures(args.manifest)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} labeled capture rows to {args.output}")


if __name__ == "__main__":
    main()
