# Places Dataset

A clean, developer‑friendly dataset of places with geographic coordinates, map links, encyclopedic context, image references, short‑term rental references, and vector embeddings for search/recommendation.

> **Columns:** `state_name`, `city_name`, `latitude`, `longitude`, `google_maps_url`, `wikipedia_content`, `unsplash_images`, `airbnb_listings`, `vector_embeddings`

---

## Quick Start

**CSV header (exact order):**

```csv
state_name,city_name,latitude,longitude,google_maps_url,wikipedia_content,unsplash_images,airbnb_listings,vector_embeddings
```

**Minimal row example:**

```csv
Rajasthan,Jaipur,26.9124,75.7873,https://www.google.com/maps?q=26.9124,75.7873,"Jaipur is the capital of the Indian state of Rajasthan...","[\"https://images.unsplash.com/photo-123\",\"https://images.unsplash.com/photo-456\"]","[{\"id\":\"jaipur-stay-1\",\"url\":\"https://airbnb.com/rooms/123\",\"price\":56,\"rating\":4.7}]","[0.121,-0.034,0.512,...]"
```

> JSON-like columns (`unsplash_images`, `airbnb_listings`, `vector_embeddings`) are stored as JSON **strings** in CSV. If you use a columnar format (Parquet/Arrow), store them as native arrays/structs.

---

## Data Dictionary

| Column              | Type                          | Required | Example                                              | Notes                                                                                                           |
| ------------------- | ----------------------------- | -------: | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `state_name`        | `string`                      |        ✓ | `Rajasthan`                                          | Top‑level administrative region. Normalize to a canonical spelling.                                             |
| `city_name`         | `string`                      |        ✓ | `Jaipur`                                             | Settlement name or locality.                                                                                    |
| `latitude`          | `number` (float)              |        ✓ | `26.9124`                                            | Decimal degrees, **WGS84**.                                                                                     |
| `longitude`         | `number` (float)              |        ✓ | `75.7873`                                            | Decimal degrees, **WGS84**.                                                                                     |
| `google_maps_url`   | `string` (URL)                |        ✓ | `https://www.google.com/maps?q=26.9124,75.7873`      | Direct place/search URL for quick preview.                                                                      |
| `wikipedia_content` | `string` (text)               |          | `"Jaipur is the capital..."`                         | Plain‑text summary or cleaned content. Keep under a sensible length (e.g., 1–3k chars) and attribute Wikipedia. |
| `unsplash_images`   | `array<string>` (JSON in CSV) |          | `["https://images.unsplash.com/...", ...]`           | One or more Unsplash image URLs relevant to the place. Respect attribution & API guidelines.                    |
| `airbnb_listings`   | `array<object>` (JSON in CSV) |          | `[{"id":"...","url":"...","price":56,"rating":4.7}]` | Optional metadata per listing: `id`, `url`, `price` (per night), `rating`, `reviews`.                           |
| `vector_embeddings` | `array<number>` (JSON in CSV) |        ✓ | `[0.121,-0.034, ...]`                                | Float vector from your embedding model (e.g., 384/768 dims). Use consistent model & dimensionality.             |

---

## JSON Schema (reference)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Place",
  "type": "object",
  "required": [
    "state_name",
    "city_name",
    "latitude",
    "longitude",
    "google_maps_url",
    "vector_embeddings"
  ],
  "properties": {
    "state_name": { "type": "string", "minLength": 1 },
    "city_name": { "type": "string", "minLength": 1 },
    "latitude": { "type": "number", "minimum": -90, "maximum": 90 },
    "longitude": { "type": "number", "minimum": -180, "maximum": 180 },
    "google_maps_url": { "type": "string", "format": "uri" },
    "wikipedia_content": { "type": "string" },
    "unsplash_images": {
      "type": "array",
      "items": { "type": "string", "format": "uri" }
    },
    "airbnb_listings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "url": { "type": "string", "format": "uri" },
          "price": { "type": "number" },
          "rating": { "type": "number", "minimum": 0, "maximum": 5 },
          "reviews": { "type": "integer", "minimum": 0 }
        }
      }
    },
    "vector_embeddings": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 2
    }
  }
}
```

---

## Usage Examples

### Python (Pandas + similarity search)

```python
import json, ast, numpy as np, pandas as pd
from numpy.linalg import norm

# Load CSV
# df = pd.read_csv("places.csv")

# Parse JSON-like columns
for col in ["unsplash_images", "airbnb_listings", "vector_embeddings"]:
    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x)

# Cosine similarity against an embedding (same dimensionality)
query_vec = np.array([0.12, -0.03, 0.51, ...])
M = np.vstack(df["vector_embeddings"].apply(np.array))
scores = (M @ query_vec) / (norm(M, axis=1) * norm(query_vec) + 1e-9)

df["similarity"] = scores
print(df.sort_values("similarity", ascending=False).head(10)[["state_name","city_name","similarity"]])
```

### SQL (PostgreSQL table sketch)

```sql
CREATE TABLE places (
  state_name       text NOT NULL,
  city_name        text NOT NULL,
  latitude         double precision NOT NULL,
  longitude        double precision NOT NULL,
  google_maps_url  text NOT NULL,
  wikipedia_content text,
  unsplash_images   jsonb,
  airbnb_listings   jsonb,
  vector_embeddings vector(768) -- use pgvector; set dimension to your model
);

-- Example: top-5 nearest neighbors with pgvector
SELECT state_name, city_name
FROM places
ORDER BY vector_embeddings <-> '[0.12,-0.03,0.51, ...]'::vector
LIMIT 5;
```

---

## Conventions & Recommendations

* **Coordinates:** WGS84 decimal degrees. Keep 5–6 decimals for city‑level precision.
* **Google Maps:** Prefer lat,lng query format or place IDs when available.
* **Wikipedia content:** Store cleaned plaintext (no markup). Keep a short summary to reduce token costs if piping to LLMs.
* **Unsplash:** Keep original URLs; record `alt_description` in your source pipeline if available.
* **Airbnb:** Only include public metadata/links you are allowed to share; respect terms of service and robots.txt.
* **Embeddings:** Use one consistent model across all rows (e.g., 384/512/768 dims). Log the model name and version separately in your pipeline metadata.

---

## Attribution & Licensing

* **Wikipedia** content is under CC BY‑SA; ensure proper attribution and share‑alike when redistributing derived text.
* **Unsplash** images require attribution when required by license/API usage; store photographer/URL if you plan to display.
* **Google Maps** links must follow Google Maps Platform Terms of Service.
* **Airbnb** data must comply with Airbnb terms; avoid scraping where prohibited.

> Choose a license appropriate for your compiled dataset (e.g., CC BY 4.0). Document any upstream license obligations.

---

## Changelog

* `v0.1.0`: Initial schema and examples.

---

## Maintainers

* *Shrey vijayvargiya* — *[shreyvijayvargiya26@gmail.com](mailto:shreyvijayvargiya26@gmail.com)*

---

## FAQ

**Why store arrays as JSON strings in CSV?**
To keep a single portable CSV while preserving rich structures. Prefer Parquet/Arrow for analytics and native arrays.

**What embedding dimension should I use?**
Whatever your model outputs. Set the `vector(dim)` accordingly in pgvector (e.g., 384/768/1024).

**Can I add more fields?**
Yes—common extensions include `country`, `district`, `population`, `timezone`, `wikipedia_url`, `unsplash_attribution`.

