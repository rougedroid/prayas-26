from copy import deepcopy

class OptimizerOutputError(ValueError):
    pass

def _coords(item, kind):
    coords = item.get("coordinates")
    if not coords:
        raise OptimizerOutputError(f"{kind} {item.get('id','<unknown>')} missing coordinates")
    return coords

def normalize_optimizer_output(result):
    if not isinstance(result, dict):
        raise OptimizerOutputError("Optimizer result must be a dictionary")
    out = {"roads": [], "zones": [], "parks": [], "pois": []}

    for r in result.get("roads", []):
        x = deepcopy(r); _coords(x, "road")
        x.setdefault("id", f"road-{len(out['roads'])+1}")
        x.setdefault("road_type", "local")
        x.setdefault("width_m", 8.0)
        out["roads"].append(x)

    for z in result.get("zones", []):
        x = deepcopy(z); _coords(x, "zone")
        x.setdefault("id", f"zone-{len(out['zones'])+1}")
        x.setdefault("zone_type", "mixed")
        x.setdefault("height_m", 0.5)
        out["zones"].append(x)

    for p in result.get("parks", []):
        x = deepcopy(p); _coords(x, "park")
        x.setdefault("id", f"park-{len(out['parks'])+1}")
        x.setdefault("height_m", 0.15)
        out["parks"].append(x)

    for p in result.get("pois", []):
        x = deepcopy(p); _coords(x, "poi")
        x.setdefault("id", f"poi-{len(out['pois'])+1}")
        x.setdefault("poi_type", "civic")
        out["pois"].append(x)

    return out
