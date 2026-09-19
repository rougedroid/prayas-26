from pathlib import Path
import json
from .adapter import normalize_optimizer_output

def _props(item, drop={"coordinates"}):
    return {k:v for k,v in item.items() if k not in drop}

def optimizer_result_to_geojson(result):
    data = normalize_optimizer_output(result)
    features = []

    for r in data["roads"]:
        p = _props(r)
        p.update({
            "feature_kind": "road",
            "olcs_altitudeMode": "clampToGround",
            "olcs_extrudedHeight": r.get("extruded_height_m", 0.05),
        })
        features.append({
            "type":"Feature", "id":r["id"],
            "geometry":{"type":"LineString","coordinates":r["coordinates"]},
            "properties":p
        })

    for z in data["zones"]:
        p = _props(z)
        p.update({
            "feature_kind":"zone",
            "olcs_altitudeMode":"clampToGround",
            "olcs_extrudedHeight": z.get("height_m", 0.5),
        })
        features.append({
            "type":"Feature","id":z["id"],
            "geometry":{"type":"Polygon","coordinates":[z["coordinates"]]},
            "properties":p
        })

    for park in data["parks"]:
        p = _props(park)
        p.update({
            "feature_kind":"park",
            "olcs_altitudeMode":"clampToGround",
            "olcs_extrudedHeight":park.get("height_m",0.15),
        })
        features.append({
            "type":"Feature","id":park["id"],
            "geometry":{"type":"Polygon","coordinates":[park["coordinates"]]},
            "properties":p
        })

    for poi in data["pois"]:
        p = _props(poi)
        p.update({"feature_kind":"poi","olcs_altitudeMode":"clampToGround"})
        if poi.get("model_url"):
            p["olcs_modelUrl"] = poi["model_url"]
            p["olcs_modelHeading"] = poi.get("model_heading",0)
            p["olcs_modelScaleX"] = poi.get("model_scale_x",1)
            p["olcs_modelScaleY"] = poi.get("model_scale_y",1)
            p["olcs_modelScaleZ"] = poi.get("model_scale_z",1)
        features.append({
            "type":"Feature","id":poi["id"],
            "geometry":{"type":"Point","coordinates":poi["coordinates"]},
            "properties":p
        })

    return {"type":"FeatureCollection","name":"prayas_optimized_city_plan","features":features}

def write_geojson(result, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(optimizer_result_to_geojson(result), indent=2), encoding="utf-8")
    return output
