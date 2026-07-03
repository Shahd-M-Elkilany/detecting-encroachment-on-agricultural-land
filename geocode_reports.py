"""
geocode_reports.py - Patch location names into KEMET1 report HTML files.

Run once after generating reports:
    python geocode_reports.py site0_report.html site3_report.html site23_report.html

Or patch all HTML reports in the current folder:
    python geocode_reports.py
"""
import sys, re, time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")

HEADERS = {"User-Agent": "KEMET1-GP-Geocoder/1.0 (graduation project)"}

def reverse_geocode(lat, lon, retries=3):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "json", "lat": lat, "lon": lon, "zoom": 14, "accept-language": "en"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            d = r.json()
            a = d.get("address", {})
            nm = (a.get("hamlet") or a.get("neighbourhood") or a.get("isolated_dwelling")
                  or a.get("village") or a.get("suburb") or a.get("quarter")
                  or a.get("allotments") or a.get("town") or a.get("municipality")
                  or a.get("city_district") or a.get("city")
                  or a.get("county") or a.get("state_district") or "")
            gv = a.get("state") or a.get("state_district") or a.get("county") or ""
            if not nm:
                pts = d.get("display_name", "").split(",")
                for p in pts:
                    p = p.strip()
                    if p and not p[0].isdigit():
                        nm = p
                        break
                if len(pts) > 1:
                    gv = pts[1].strip()
            label = nm or "Nile Delta area"
            if gv and gv != label:
                label += " – " + gv
            return label
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None

def patch_html(path: Path):
    html = path.read_text(encoding="utf-8")

    # Find all rows: data-ci, data-lat, data-lon
    rows = re.findall(
        r"data-ci='(\d+)'\s+data-lat='([\d.\-]+)'\s+data-lon='([\d.\-]+)'",
        html
    )
    if not rows:
        print(f"  No clusters found in {path.name}, skipping.")
        return

    print(f"\n{path.name}: {len(rows)} clusters")
    replacements = {}
    for ci, lat, lon in rows:
        print(f"  Cluster #{int(ci)+1} ({lat}N {lon}E) ... ", end="", flush=True)
        name = reverse_geocode(lat, lon)
        if name:
            print(name)
            replacements[ci] = name
        else:
            print("(failed, keeping loading...)")
        time.sleep(1.1)  # Nominatim rate limit: max 1 req/sec

    # Replace each span: <span data-name='N' ...>loading...</span>
    def replacer(m):
        ci = m.group(1)
        if ci in replacements:
            return f"<span data-name='{ci}' style='color:#c9d1d9'>{replacements[ci]}</span>"
        return m.group(0)

    patched = re.sub(
        r"<span data-name='(\d+)'[^>]*>.*?</span>",
        replacer,
        html
    )

    # Remove the geocoding JS block — no longer needed
    patched = re.sub(r"<script>\s*function _showName[\s\S]*?</script>", "", patched)
    patched = re.sub(r"<script>\s*function _jsonpGeo[\s\S]*?</script>", "", patched)
    patched = re.sub(r"<script>\s*function _parseAddr[\s\S]*?</script>", "", patched)
    patched = re.sub(r"<script>\s*function _fetchGeo[\s\S]*?</script>", "", patched)

    path.write_text(patched, encoding="utf-8")
    print(f"  Saved: {path.name}")

def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(Path(".").glob("*_report.html"))

    if not targets:
        sys.exit("No HTML report files found.")

    print(f"Geocoding {len(targets)} report(s)...")
    for p in targets:
        if p.exists():
            patch_html(p)
        else:
            print(f"Not found: {p}")

    print("\nDone. Reload the HTML files in your browser.")

if __name__ == "__main__":
    main()
