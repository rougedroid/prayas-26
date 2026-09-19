from neo4j import GraphDatabase


class Neo4jLoader:

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        self.database = database

        self.driver.verify_connectivity()

    def load_cells(self):
        query = """
        MATCH (c:Cell)
        OPTIONAL MATCH (c)-[:ADJACENT]-(n)

        RETURN
            c.cell_id AS id,
            c.population AS population,
            c.type AS type,
            c.green_cover_pct AS green_cover,
            c.elevation_m AS elevation,
            c.dist_to_boundary_m AS distance_to_boundary,
            c.x AS x,
            c.y AS y,
            collect(DISTINCT n.cell_id) AS neighbors
        ORDER BY c.cell_id
        """

        records, _, _ = self.driver.execute_query(
            query,
            database_=self.database,
        )

        return [record.data() for record in records]

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
