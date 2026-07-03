// ============================================================
//  PASTE THIS INTO: https://code.earthengine.google.com
//  Click ▶ Run  →  then go to Tasks tab → click Run on each
//
//  Downloads 5 sample Nile Delta tiles for 2021 and 2023.
//  Each tile = same geographic location both years.
//  Output: Google Drive → folder "NileDelta_samples"
//  Format: GeoTIFF, 6 spectral index bands, 10 m/px, 260×260 px
//  Bands:  1=NDVI  2=NDBI  3=MNDWI  4=SAVI  5=BSI  6=NDWI
// ============================================================

// ── 5 sample locations (known encroachment fringes) ──────────
var SAMPLES = [
  {name: 'Tanta_fringe',      lon: 31.002, lat: 30.794},
  {name: 'Mansoura_fringe',   lon: 31.378, lat: 31.037},
  {name: 'Damanhur_fringe',   lon: 30.468, lat: 30.468},
  {name: 'Zagazig_fringe',    lon: 31.502, lat: 30.584},
  {name: 'Kafr_El_Sheikh',    lon: 30.935, lat: 31.113},
];

var YEARS        = [2021, 2023, 2025];
var SEASON_START = '-07-01';
var SEASON_END   = '-09-15';
var HALF_DEG     = 0.012;   // ~1.3 km → 2.6 km tile
var DRIVE_FOLDER = 'NileDelta_samples';

// ── Spectral index computation ────────────────────────────────
function addIndices(img) {
  var B2  = img.select('B2');
  var B3  = img.select('B3');
  var B4  = img.select('B4');
  var B8  = img.select('B8');
  var B11 = img.select('B11');
  var eps = 1e-6;

  var NDVI  = B8.subtract(B4).divide(B8.add(B4).add(eps)).rename('NDVI');
  var NDBI  = B11.subtract(B8).divide(B11.add(B8).add(eps)).rename('NDBI');
  var MNDWI = B3.subtract(B11).divide(B3.add(B11).add(eps)).rename('MNDWI');
  var SAVI  = B8.subtract(B4).multiply(1.5)
               .divide(B8.add(B4).add(0.5).add(eps)).rename('SAVI');
  var BSI   = B11.add(B4).subtract(B8.add(B2))
               .divide(B11.add(B4).add(B8).add(B2).add(eps)).rename('BSI');
  var NDWI  = B3.subtract(B8).divide(B3.add(B8).add(eps)).rename('NDWI');

  return ee.Image([NDVI, NDBI, MNDWI, SAVI, BSI, NDWI]);
}

// ── Cloud masking ─────────────────────────────────────────────
function maskClouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(7)).and(scl.neq(8))
               .and(scl.neq(9)).and(scl.neq(10));
  return img.updateMask(mask).divide(10000);
}

// ── Submit export for one location + year ─────────────────────
function exportTile(sample, year) {
  var roi = ee.Geometry.Rectangle([
    sample.lon - HALF_DEG, sample.lat - HALF_DEG,
    sample.lon + HALF_DEG, sample.lat + HALF_DEG
  ]);

  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(year + SEASON_START, year + SEASON_END)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskClouds)
    .select(['B2','B3','B4','B8','B11','SCL']);

  var composite = col.median();
  var indices   = addIndices(composite).clip(roi);

  var taskName  = sample.name + '_' + year;

  Export.image.toDrive({
    image:          indices,
    description:    taskName,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: taskName,
    region:         roi,
    scale:          10,
    crs:            'EPSG:4326',
    maxPixels:      1e8,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + taskName);
}

// ── Visualise on map ──────────────────────────────────────────
Map.setCenter(31.0, 30.8, 9);
Map.setOptions('SATELLITE');

// Show all sample locations on the map
SAMPLES.forEach(function(s) {
  var roi = ee.Geometry.Rectangle([
    s.lon - HALF_DEG, s.lat - HALF_DEG,
    s.lon + HALF_DEG, s.lat + HALF_DEG
  ]);
  Map.addLayer(roi, {color: 'FF4444'}, s.name);

  // Also show 2021 NDVI on the map for first sample
  if (s.name === 'Tanta_fringe') {
    var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi)
      .filterDate('2021-07-01', '2021-09-15')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(maskClouds)
      .select(['B2','B3','B4','B8','B11','SCL'])
      .median();
    var ndvi = col.normalizedDifference(['B8','B4']).rename('NDVI').clip(roi);
    Map.addLayer(ndvi, {min:0, max:0.7, palette:['brown','yellow','green']},
                 'Tanta NDVI 2021');
  }
});

// ── Submit all export tasks ───────────────────────────────────
SAMPLES.forEach(function(sample) {
  YEARS.forEach(function(year) {
    exportTile(sample, year);
  });
});

print('');
print('✅ ' + (SAMPLES.length * YEARS.length) + ' tasks queued.');
print('Go to the Tasks tab (top-right) and click Run on each task.');
print('Files will appear in Google Drive → ' + DRIVE_FOLDER);
