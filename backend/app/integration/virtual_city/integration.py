from pathlib import Path
from .geojson import write_geojson
from .publisher import VCPublisherClient

def export_optimizer_result(optimizer_result, output_dir="artifacts/virtual_city",
                            filename="prayas_optimized_city.geojson"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_geojson(optimizer_result, output_dir / filename)

def publish_optimizer_result(optimizer_result, config,
                             output_dir="artifacts/virtual_city",
                             datasource_name="Prayas Optimized City Plan"):
    path = export_optimizer_result(optimizer_result, output_dir=output_dir)
    datasource = VCPublisherClient(config).publish_geojson(
        path, datasource_name=datasource_name
    )
    return {"local_geojson":str(path), "datasource":datasource}
