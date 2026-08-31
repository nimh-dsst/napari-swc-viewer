#!/usr/bin/env python
"""Download the CPD2 registered SWC files from Brain Image Library."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import ssl
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit


BIL_DOI_URL = "https://doi.brainimagelibrary.org/doi/10.35077/g.73"
USER_AGENT = "napari-neuron-navigator-cpd2-downloader/1.0"

# Generated from the current cpd2.parquet file_id values by replacing the
# locally aligned "_reg_right.swc" suffix with the BIL source "_reg.swc" suffix.
CPD2_REGISTERED_FILENAMES = (
    "1119749422_17302_1892-X17514-Y37780_reg.swc",
    "1119749431_17302_2332-X18576-Y20014_reg.swc",
    "1119749441_17302_2416-X18816-Y39212_reg.swc",
    "1119749447_17302_2595-X14957-Y42174_reg.swc",
    "1119749665_17545_3134-X21894-Y19320_reg.swc",
    "1119749669_17545_3182-X21534-Y14784_reg.swc",
    "1119749693_17545_3521-X23923-Y36782_reg.swc",
    "1119749714_17545_3601-X21497-Y38754_reg.swc",
    "1119749730_17545_3678-X20788-Y38882_reg.swc",
    "1119749736_17545_3738-X20745-Y38576_reg.swc",
    "1119749740_17545_3781-X18263-Y13599_reg.swc",
    "1119749787_17545_5238-X18956-Y11404_reg.swc",
    "1119749812_17545_5494-X19511-Y10228_reg.swc",
    "1119749818_17545_5564-X19752-Y10120_reg.swc",
    "1119749947_17787_3772-X13169-Y10621_reg.swc",
    "1119750305_18454_4053-X8213-Y10031_reg.swc",
    "1119750739_18455_3774-X27258-Y10252_reg.swc",
    "1119750754_18455_4028-X31096-Y13276_reg.swc",
    "1119750782_18455_5058-X31546-Y10774_reg.swc",
    "1119750897_18455_5678-X8306-Y13292_reg.swc",
    "1119750955_18457_2987-X24716-Y11295_reg.swc",
    "1119750963_18457_3135-X25362-Y10752_reg.swc",
    "1119750965_18457_3199-X14932-Y11625_reg.swc",
    "1119750972_18457_3327-X14429-Y13609_reg.swc",
    "1119750976_18457_3363-X27428-Y14011_reg.swc",
    "1119750985_18457_3461-X24104-Y11097_reg.swc",
    "1119750991_18457_3541-X14011-Y10702_reg.swc",
    "1119750993_18457_3602-X26510-Y13816_reg.swc",
    "1119750996_18457_3624-X25725-Y9288_reg.swc",
    "1119750998_18457_3631-X25758-Y11819_reg.swc",
    "1119751000_18457_3695-X24888-Y9339_reg.swc",
    "1119751002_18457_3811-X12535-Y8935_reg.swc",
    "1119751006_18457_3868-X12980-Y11370_reg.swc",
    "1119751008_18457_3936-X23836-Y9498_reg.swc",
    "1119751014_18457_3995-X28318-Y13066_reg.swc",
    "1119751016_18457_3998-X24655-Y9472_reg.swc",
    "1119751026_18457_4267-X24842-Y9528_reg.swc",
    "1119751028_18457_4293-X12802-Y8782_reg.swc",
    "1119751032_18457_4330-X26376-Y9756_reg.swc",
    "1119751034_18457_4363-X26172-Y9975_reg.swc",
    "1119751242_18458_3970-X10955-Y15932_reg.swc",
    "1119751244_18458_3988-X12064-Y15356_reg.swc",
    "1119751614_18463_5932-X25452-Y10372_reg.swc",
    "1119751652_18464_3690-X11378-Y11250_reg.swc",
    "1119751654_18464_3775-X9476-Y10896_reg.swc",
    "1119751656_18464_3803-X7165-Y13366_reg.swc",
    "1119751662_18464_3913-X6367-Y16275_reg.swc",
    "1119751668_18464_4092-X22324-Y12272_reg.swc",
    "1119751671_18464_4105-X26619-Y16058_reg.swc",
    "1119751795_18465_3136-X13799-Y9638_reg.swc",
    "1119751801_18465_3342-X13698-Y10751_reg.swc",
    "1119751810_18465_3562-X11349-Y11115_reg.swc",
    "1119751812_18465_3744-X7900-Y14714_reg.swc",
    "1119751816_18465_3753-X26510-Y15402_reg.swc",
    "1119751818_18465_3785-X12414-Y8456_reg.swc",
    "1119751826_18465_3961-X25007-Y12191_reg.swc",
    "1119751830_18465_4006-X11605-Y12004_reg.swc",
    "1119751832_18465_4025-X7485-Y13129_reg.swc",
    "1119751834_18465_4064-X27958-Y16140_reg.swc",
    "1119751838_18465_4134-X10308-Y9775_reg.swc",
    "1119751840_18465_4138-X11465-Y9479_reg.swc",
    "1119751842_18465_4251-X9022-Y9409_reg.swc",
    "1119751846_18465_4306-X9585-Y11469_reg.swc",
    "1119751848_18465_4332-X11122-Y8645_reg.swc",
    "1119751851_18465_4352-X6500-Y14855_reg.swc",
    "1119751853_18465_4475-X7589-Y12842_reg.swc",
    "1119751855_18465_4644-X30647-Y14574_reg.swc",
    "1119751858_18465_4738-X7182-Y12251_reg.swc",
    "1119751860_18465_4759-X6422-Y14561_reg.swc",
    "1119751864_18465_4822-X8699-Y9995_reg.swc",
    "1119751868_18465_4875-X7054-Y11860_reg.swc",
    "1119751870_18465_4875-X8909-Y10214_reg.swc",
    "1119751872_18465_4927-X7645-Y11238_reg.swc",
    "1119751883_18465_5019-X7656-Y9722_reg.swc",
    "1119751891_18465_5121-X8103-Y10000_reg.swc",
    "1119751893_18465_5146-X7379-Y10539_reg.swc",
    "1119751933_18465_5293-X30583-Y13343_reg.swc",
)


def make_ssl_context(verify_ssl: bool) -> ssl.SSLContext:
    """Return the SSL context used for BIL requests."""
    if verify_ssl:
        return ssl.create_default_context()

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def request_url(url: str) -> urllib.request.Request:
    """Build a BIL HTTP request with a stable user agent."""
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def read_url_text(url: str, context: ssl.SSLContext, timeout: float) -> str:
    """Read one URL as text."""
    request = request_url(url)
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def corrected_registered_filename(filename: str) -> str:
    """Return the BIL registered source filename for a CPD2 parquet filename."""
    name = Path(filename).name
    if name.endswith("_reg_right.swc"):
        return name.removesuffix("_reg_right.swc") + "_reg.swc"
    return name


def extract_swc_urls(markup: str) -> list[str]:
    """Extract unique BIL SWC download URLs from a DOI landing page."""
    unescaped = html.unescape(markup)
    pattern = re.compile(
        r"https://download\.brainimagelibrary\.org/[^\s\"'<>]+?\.swc"
    )
    urls: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(unescaped):
        url = match.group(0)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def filename_from_url(url: str) -> str:
    """Return the final path component from a download URL."""
    return unquote(Path(urlsplit(url).path).name)


def match_target_urls(
    target_filenames: tuple[str, ...],
    swc_urls: list[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Match target filenames to DOI-page URLs by basename."""
    urls_by_name = {filename_from_url(url): url for url in swc_urls}
    matched: list[tuple[str, str]] = []
    missing: list[str] = []

    for filename in target_filenames:
        url = urls_by_name.get(filename)
        if url is None:
            missing.append(filename)
        else:
            matched.append((filename, url))

    return matched, missing


def read_targets_from_parquet(parquet_path: Path) -> tuple[str, ...]:
    """Read target SWC filenames from an existing CPD2 parquet file."""
    import pandas as pd

    file_ids = pd.read_parquet(parquet_path, columns=["file_id"])["file_id"]
    return tuple(
        corrected_registered_filename(str(file_id))
        for file_id in file_ids.dropna().drop_duplicates()
    )


def resolve_cpd2_urls(
    *,
    doi_url: str,
    target_filenames: tuple[str, ...],
    context: ssl.SSLContext,
    timeout: float,
) -> list[tuple[str, str]]:
    """Resolve the CPD2 target filenames to concrete BIL download URLs."""
    markup = read_url_text(doi_url, context, timeout)
    swc_urls = extract_swc_urls(markup)
    matched, missing = match_target_urls(target_filenames, swc_urls)
    if missing:
        sample = "\n".join(f"  - {name}" for name in missing[:10])
        extra = "" if len(missing) <= 10 else f"\n  ... {len(missing) - 10} more"
        raise RuntimeError(
            f"Could not find {len(missing)} CPD2 file(s) in {doi_url}:\n"
            f"{sample}{extra}"
        )
    return matched


def write_manifest(matches: list[tuple[str, str]], manifest_path: Path) -> None:
    """Write a tab-separated filename-to-URL manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["filename\turl"]
    lines.extend(f"{filename}\t{url}" for filename, url in matches)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_file(
    url: str,
    output_path: Path,
    *,
    context: ssl.SSLContext,
    timeout: float,
    force: bool,
) -> str:
    """Download one file, returning a status string."""
    if output_path.exists() and not force:
        return "exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    request = request_url(url)
    try:
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            with tmp_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return "downloaded"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the 77 registered CPD2 SWC files from the Brain Image "
            "Library dataset DOI page."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cpd2_data"),
        help="Directory where SWC files will be written. Default: cpd2_data",
    )
    parser.add_argument(
        "--doi-url",
        default=BIL_DOI_URL,
        help=f"BIL DOI landing page to parse. Default: {BIL_DOI_URL}",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help=(
            "Read target names from an existing CPD2 parquet file instead of "
            "the built-in 77-file list."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write a TSV manifest of resolved filenames and URLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the CPD2 URLs without downloading files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files that already exist in the output directory.",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Verify BIL SSL certificates. Disabled by default to match BIL scripts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds. Default: 60",
    )
    return parser.parse_args()


def main() -> int:
    """Run the downloader."""
    parsed = parse_args()
    context = make_ssl_context(parsed.verify_ssl)
    target_filenames = (
        read_targets_from_parquet(parsed.parquet)
        if parsed.parquet is not None
        else CPD2_REGISTERED_FILENAMES
    )

    print(f"Resolving {len(target_filenames)} CPD2 files from {parsed.doi_url}")
    matches = resolve_cpd2_urls(
        doi_url=parsed.doi_url,
        target_filenames=target_filenames,
        context=context,
        timeout=parsed.timeout,
    )
    print(f"Matched {len(matches)} CPD2 registered SWC files.")

    if parsed.manifest is not None:
        write_manifest(matches, parsed.manifest)
        print(f"Wrote manifest to {parsed.manifest}")

    if parsed.dry_run:
        for filename, url in matches:
            print(f"{filename}\t{url}")
        return 0

    downloaded = 0
    skipped = 0
    failed = 0
    for index, (filename, url) in enumerate(matches, start=1):
        output_path = parsed.output_dir / filename
        try:
            status = download_file(
                url,
                output_path,
                context=context,
                timeout=parsed.timeout,
                force=parsed.force,
            )
        except Exception as exc:
            failed += 1
            print(f"[{index:02d}/{len(matches)}] failed {filename}: {exc}")
            continue

        if status == "exists":
            skipped += 1
            print(f"[{index:02d}/{len(matches)}] exists {filename}")
        else:
            downloaded += 1
            print(f"[{index:02d}/{len(matches)}] downloaded {filename}")

    print(
        f"Done: {downloaded} downloaded, {skipped} already present, "
        f"{failed} failed in {parsed.output_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
